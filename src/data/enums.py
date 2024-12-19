from enum import Enum


class TypeToken(Enum):
    MEDICATION = "[-MED-]"
    EVENTS = "[-EVT-]"
    FLUID_IN = "[-FLI-]"
    FLUID_OUT = "[-FLO-]"
    LDA = "[-LDA-]"
    MEASUREMENTS = "[-MEA-]"
    ORDERS = "[-ORD-]"
    RANGE_SIGNALS_JOINED = "[-RSJ-]"
    LABORATORY = "[-LAB-]"
    FLUID_BALANCE = "[-FLB-]"
    HEMODYNAMICS = "[-HEM-]"
    INFECTIOLOGY = "[-INF-]"
    RESPIRATORY = "[-RES-]"
    NEUROLOGY = "[-NEU-]"
    RENAL_REPLACEMENT_THERAPY = "[-RRT-]"
    ADMISSION_INFORMATION = "[-ADI-]"
    CLINICAL_SCORE = "[-CLI-]"
    NICE_DATA = "[-NIC-]"
    DEMOGRAPHICS = "[-DEM-]"
    POSITION = "[-POS-]"
    SOFA_SCORE = "[-SOF-]"
    OTHERS = "[-OTH-]"


class EHRToken(Enum):
    EPIC = "[-EPIC-]"
    HIX = "[-HIX-]"
    MV = "[-MV-]"


class VocabularyToken(Enum):
    RXNORM = "[-RXNORM-]"
    LOINC = "[-LOINC-]"
    ICUDATA = "[-ICUDATA-]"
    ICUNITY = "[-ICUNITY-]"
    ATC = "[-ATC-]"


class PoolingMode(Enum):
    MEAN = "mean"
    CLS = "cls"
