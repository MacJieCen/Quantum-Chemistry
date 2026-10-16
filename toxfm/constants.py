"""Stable endpoint names shared by data preparation, training and prediction."""

ENDPOINT_FILES = {
    "FirstStage": "FirstStage.csv",
    "Hemolytic": "SecondStage_Hemolytic.csv",
    "Hematotoxicity": "SecondStage_Hematotoxic.csv",
    "Myelotoxicity": "SecondStage_Myelotoxicity.csv",
    "Thrombocytopenia": "SecondStage_Thrombocytopenia.csv",
}
SUBTYPES = tuple(x for x in ENDPOINT_FILES if x != "FirstStage")
GRAPH_SCHEMA = "simg-x39-edge27-v1"


def endpoint_name(value):
    aliases = {x.lower(): x for x in ENDPOINT_FILES}
    aliases.update(hematotoxic="Hematotoxicity", myelotoxic="Myelotoxicity",
                   hemolytictoxicity="Hemolytic", firststage="FirstStage", all="all")
    try:
        return aliases[value.lower().replace("-", "").replace("_", "")]
    except KeyError:
        raise ValueError("Unknown toxicity type {!r}; choose {} or all".format(value, ", ".join(SUBTYPES)))
