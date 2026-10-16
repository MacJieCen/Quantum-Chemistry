"""SIMG* graph encoder and node-preserving ParseCaps-inspired routing.

Extracted from the authors server implementation without changing model math.
This graph adaptation does not implement spatial PConvCaps or alpha-entmax.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.utils import to_dense_batch

ARCHITECTURE = "simg_graph_parsecaps_dynamic_v2"

def squash(x, eps=1e-8):
    norm = torch.norm(x, dim=-1, keepdim=True)
    return (norm.square() / (1.0 + norm.square())) * (x / (norm + eps))


def capsule_logits(norms):
    return torch.log(norms[:, 1].clamp_min(1e-8)) - torch.log(norms[:, 0].clamp_min(1e-8))


class InteractionMessageBlock(nn.Module):
    """Distance-gated messages over SIMG's precomputed interaction edges."""
    def __init__(self, hidden_dim, dropout, rbf_bins=16):
        super(InteractionMessageBlock, self).__init__()
        self.register_buffer("centers", torch.linspace(0.0, 6.0, rbf_bins))
        self.gamma = 2.0
        self.message = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(rbf_bins, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = float(dropout)

    def forward(self, hidden, interaction_edge_index, coordinates):
        if interaction_edge_index.numel() == 0:
            return hidden
        source, target = interaction_edge_index[0], interaction_edge_index[1]
        distance = torch.norm(coordinates[source] - coordinates[target], dim=-1, keepdim=True)
        rbf = torch.exp(-self.gamma * (distance - self.centers.view(1, -1)).square())
        messages = self.message(hidden[source]) * self.gate(rbf)
        aggregate = torch.zeros_like(hidden)
        aggregate.index_add_(0, target, messages)
        degree = torch.zeros((hidden.size(0), 1), dtype=hidden.dtype, device=hidden.device)
        degree.index_add_(0, target, torch.ones((target.numel(), 1), dtype=hidden.dtype, device=hidden.device))
        updated = self.norm(hidden + aggregate / degree.clamp_min(1.0))
        return F.dropout(F.gelu(updated), p=self.dropout, training=self.training)


class TypeAwareInteractionGINEEncoder(nn.Module):
    """SIMG covalent GINE encoder with optional native type and interaction data."""
    def __init__(self, input_dim, edge_dim, hidden_dim, num_layers, dropout,
                 use_node_types, use_interactions, use_multiscale):
        super(TypeAwareInteractionGINEEncoder, self).__init__()
        self.use_node_types = bool(use_node_types)
        self.use_interactions = bool(use_interactions)
        self.use_multiscale = bool(use_multiscale)
        self.node_encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.edge_encoder = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.type_embedding = nn.Embedding(3, hidden_dim) if self.use_node_types else None
        self.convs, self.norms, self.interactions = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.dropout = float(dropout)
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
            if self.use_interactions:
                self.interactions.append(InteractionMessageBlock(hidden_dim, dropout))
        self.multiscale_projection = nn.Sequential(
            nn.Linear(hidden_dim * (num_layers + 1), hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        ) if self.use_multiscale else None

    @staticmethod
    def node_types(data):
        if not all(hasattr(data, name) for name in ("is_atom", "is_lp", "is_bond")):
            raise ValueError("SIMG graph is missing atom/lone-pair/bond indicators")
        membership = torch.stack((data.is_atom, data.is_lp, data.is_bond), dim=-1)
        if not bool((membership.sum(dim=-1) == 1).all()):
            raise ValueError("Every SIMG node must have exactly one node-type indicator")
        return membership.argmax(dim=-1).long()

    def forward(self, data, x):
        hidden = self.node_encoder(x)
        if self.type_embedding is not None:
            hidden = hidden + self.type_embedding(self.node_types(data))
        edge_embedding = self.edge_encoder(data.edge_attr)
        scales = [hidden]
        for layer_index, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            residual = hidden
            hidden = conv(hidden, data.edge_index, edge_embedding)
            hidden = F.gelu(norm(hidden + residual))
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            if self.use_interactions:
                if not hasattr(data, "interaction_edge_index") or not hasattr(data, "xyz_data"):
                    raise ValueError("Interaction candidate requires SIMG interaction edges and coordinates")
                hidden = self.interactions[layer_index](hidden, data.interaction_edge_index, data.xyz_data.float())
            scales.append(hidden)
        if self.multiscale_projection is not None:
            hidden = self.multiscale_projection(torch.cat(scales, dim=-1))
        return hidden


class NodePrimaryCapsules(nn.Module):
    def __init__(self, hidden_dim, num_primary, capsule_dim, dropout, use_node_types):
        super(NodePrimaryCapsules, self).__init__()
        self.num_primary = int(num_primary)
        self.capsule_dim = int(capsule_dim)
        self.use_node_types = bool(use_node_types)
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_primary * capsule_dim)
        )
        self.type_bias = nn.Embedding(3, num_primary * capsule_dim) if self.use_node_types else None

    def forward(self, hidden, node_types=None):
        tokens = self.project(hidden)
        if self.type_bias is not None:
            if node_types is None:
                raise ValueError("Type-aware primary capsules require node types")
            tokens = tokens + self.type_bias(node_types)
        return squash(tokens.view(hidden.size(0), hidden.size(1), self.num_primary, self.capsule_dim))


class GraphDynamicParseCapsRouting(nn.Module):
    """Graph adaptation of ParseCaps part-whole routing with iterative agreement."""
    def __init__(self, num_primary, capsule_dim, num_classes, iterations):
        super(GraphDynamicParseCapsRouting, self).__init__()
        self.num_primary = int(num_primary)
        self.capsule_dim = int(capsule_dim)
        self.num_classes = int(num_classes)
        self.iterations = int(iterations)
        if self.iterations < 1:
            raise ValueError("routing iterations must be positive")
        transforms = torch.empty(num_primary, num_classes, capsule_dim, capsule_dim)
        self.transforms = nn.Parameter(nn.init.xavier_uniform_(transforms))
        self.class_query = nn.Parameter(torch.empty(num_classes, capsule_dim))
        nn.init.normal_(self.class_query, std=0.02)

    def forward(self, primary, node_mask, fixed_routing_keep=None):
        if fixed_routing_keep is not None:
            raise ValueError("Dynamic routing does not accept frozen attention masks")
        # [B, N, P, D] x [P, K, D, D] -> [B, N, P, K, D]
        votes = torch.einsum("bnpd,pkde->bnpke", primary, self.transforms)
        logits = (votes * self.class_query.view(1, 1, 1, self.num_classes, self.capsule_dim)).sum(dim=-1)
        valid_children = node_mask.unsqueeze(-1).expand(-1, -1, self.num_primary)
        valid_float = valid_children.unsqueeze(-1).float()
        class_capsules = None
        coupling = None
        for iteration in range(self.iterations):
            coupling = F.softmax(logits, dim=-1) * valid_float
            child_count = valid_float.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
            preactivation = (coupling.unsqueeze(-1) * votes).sum(dim=(1, 2)) / child_count.squeeze(2)
            class_capsules = squash(preactivation)
            if iteration + 1 < self.iterations:
                agreement = (votes * class_capsules.unsqueeze(1).unsqueeze(1)).sum(dim=-1)
                logits = logits + agreement * valid_float
        return class_capsules, coupling, valid_children


class DescriptorResidualFusion(nn.Module):
    """Training-fold standardized RDKit descriptors injected as graph residuals."""
    def __init__(self, descriptor_dim, hidden_dim, dropout):
        super(DescriptorResidualFusion, self).__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(descriptor_dim), nn.Linear(descriptor_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())

    def forward(self, hidden, descriptor, batch_index):
        context = self.network(descriptor)
        return hidden + self.gate(context)[batch_index] * context[batch_index]


class FormalSIMGParseCaps(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, capsule_dim, num_primary,
                 num_layers, routing_iterations, dropout, descriptor_dim=0, num_classes=2,
                 use_node_types=False, use_interactions=False, use_multiscale=False):
        super(FormalSIMGParseCaps, self).__init__()
        self.num_primary = int(num_primary)
        self.num_classes = int(num_classes)
        self.descriptor_dim = int(descriptor_dim)
        self.use_node_types = bool(use_node_types)
        self.encoder = TypeAwareInteractionGINEEncoder(
            input_dim, edge_dim, hidden_dim, num_layers, dropout, use_node_types, use_interactions, use_multiscale
        )
        self.descriptor_fusion = DescriptorResidualFusion(descriptor_dim, hidden_dim, dropout) if descriptor_dim > 0 else None
        self.primary = NodePrimaryCapsules(hidden_dim, num_primary, capsule_dim, dropout, use_node_types)
        self.routing = GraphDynamicParseCapsRouting(num_primary, capsule_dim, num_classes, routing_iterations)

    def forward(self, data, input_node_mask=None, fixed_routing_keep=None, return_routing=False):
        x = data.x
        if input_node_mask is not None:
            if input_node_mask.ndim != 1 or input_node_mask.numel() != x.size(0):
                raise ValueError("input_node_mask must align with flattened graph nodes")
            x = x * input_node_mask.to(dtype=x.dtype, device=x.device).unsqueeze(-1)
        hidden = self.encoder(data, x)
        if self.descriptor_fusion is not None:
            if not hasattr(data, "descriptor"):
                raise ValueError("Descriptor-fusion model received a batch without descriptors")
            hidden = self.descriptor_fusion(hidden, data.descriptor.float(), data.batch)
        dense_hidden, node_mask = to_dense_batch(hidden, data.batch)
        dense_types = None
        if self.use_node_types:
            dense_types, type_mask = to_dense_batch(self.encoder.node_types(data), data.batch)
            if not bool(torch.equal(node_mask, type_mask)):
                raise RuntimeError("Dense node and type masks disagree")
        primary = self.primary(dense_hidden, dense_types)
        class_capsules, coupling, valid_children = self.routing(primary, node_mask, fixed_routing_keep)
        norms = torch.norm(class_capsules, dim=-1)
        scores = torch.sigmoid(capsule_logits(norms))
        if not return_routing:
            return norms, scores
        node_routing = coupling[:, :, :, 1].sum(dim=-1)
        return norms, scores, {
            "node_routing": node_routing,
            "node_mask": node_mask,
            "coupling": coupling,
            "routing_keep": valid_children,
        }


class GraphHouseholderParseCapsRouting(nn.Module):
    """ParseCaps-inspired orthogonal attention followed by node agreement.

    Upstream ParseCaps obtains attention weights from Householder transforms of
    spatial capsule maps. Here every class and capsule dimension has a product
    of Householder reflections over the primary-capsule-type axis. This is
    permutation independent over molecular nodes and remains exactly
    orthogonal (up to floating-point precision).
    """

    def __init__(self, num_primary, capsule_dim, num_classes, iterations, reflections):
        super().__init__()
        self.num_primary = int(num_primary)
        self.capsule_dim = int(capsule_dim)
        self.num_classes = int(num_classes)
        self.iterations = int(iterations)
        self.reflections = int(reflections)
        if self.iterations < 1 or self.reflections < 1:
            raise ValueError("routing iterations and reflections must be positive")

        transforms = torch.empty(num_primary, num_classes, capsule_dim, capsule_dim)
        self.transforms = nn.Parameter(nn.init.xavier_uniform_(transforms))
        self.class_query = nn.Parameter(torch.empty(num_classes, capsule_dim))
        nn.init.normal_(self.class_query, std=0.02)
        vectors = torch.empty(num_classes, capsule_dim, reflections, num_primary)
        self.householder_vectors = nn.Parameter(nn.init.normal_(vectors, std=0.02))

    def orthogonal_mixing(self):
        """Return [class, dimension, output-primary, input-primary] matrices."""
        eye = torch.eye(
            self.num_primary,
            dtype=self.householder_vectors.dtype,
            device=self.householder_vectors.device,
        )
        mixing = eye.view(1, 1, self.num_primary, self.num_primary).expand(
            self.num_classes, self.capsule_dim, -1, -1
        )
        for reflection_index in range(self.reflections):
            vector = self.householder_vectors[:, :, reflection_index, :]
            vector = F.normalize(vector, dim=-1, eps=1e-8)
            reflector = eye.view(1, 1, self.num_primary, self.num_primary) - 2.0 * torch.einsum(
                "kdp,kdq->kdpq", vector, vector
            )
            mixing = torch.matmul(reflector, mixing)
        return mixing

    def forward(self, primary, node_mask, fixed_routing_keep=None):
        if fixed_routing_keep is not None:
            raise ValueError("Householder routing does not accept frozen attention masks")
        # Householder attention across primary types, never across arbitrary node order.
        mixing = self.orthogonal_mixing()
        attention_logits = torch.einsum("bnpd,kdqp->bnkqd", primary, mixing)
        attention = F.softmax(attention_logits / math.sqrt(float(self.num_primary)), dim=3)
        attended = primary.unsqueeze(3) * attention.permute(0, 1, 3, 2, 4)
        # [B,N,P,K,D] x [P,K,D,D] -> one vote for every primary child and class.
        votes = torch.einsum("bnpkd,pkde->bnpke", attended, self.transforms)
        logits = (votes * self.class_query.view(1, 1, 1, self.num_classes, self.capsule_dim)).sum(dim=-1)
        valid_children = node_mask.unsqueeze(-1).expand(-1, -1, self.num_primary)
        valid_float = valid_children.unsqueeze(-1).float()
        class_capsules = None
        coupling = None
        for iteration in range(self.iterations):
            coupling = F.softmax(logits, dim=-1) * valid_float
            child_count = valid_float.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
            preactivation = (coupling.unsqueeze(-1) * votes).sum(dim=(1, 2)) / child_count.squeeze(2)
            class_capsules = squash(preactivation)
            if iteration + 1 < self.iterations:
                agreement = (votes * class_capsules.unsqueeze(1).unsqueeze(1)).sum(dim=-1)
                logits = logits + agreement * valid_float
        return class_capsules, coupling, valid_children


class FormalSIMGHouseholderParseCaps(FormalSIMGParseCaps):
    """v2 SIMG encoder with optional graph-native ParseCaps orthogonal routing."""

    def __init__(self, *args, use_orthogonal_attention=False, orthogonal_reflections=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_orthogonal_attention = bool(use_orthogonal_attention)
        self.orthogonal_reflections = int(orthogonal_reflections)
        if self.use_orthogonal_attention:
            self.routing = GraphHouseholderParseCapsRouting(
                self.num_primary,
                self.routing.capsule_dim,
                self.num_classes,
                self.routing.iterations,
                self.orthogonal_reflections,
            )
