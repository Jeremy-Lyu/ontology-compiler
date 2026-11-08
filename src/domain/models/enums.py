"""Core Enums - Shared enumeration types for ontology processing."""

from enum import Enum


class LayerType(str, Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L2_5 = "L2.5"
    L3 = "L3"
    L0_RAW_EVIDENCE = "L0"
    L1_CANDIDATE = "L1"
    L2_CANONICAL = "L2"
    L2_5_SEMANTIC = "L2.5"
    L3_CROSS_FILE = "L3"
    BUSINESS = "L1"
    CONTRACT = "L2"
    TECHNICAL = "L3"
    OPERATIONAL = "L2.5"


class OntologyRelationType(str, Enum):
    CONTAINS = "CONTAINS"
    COMPRISES = "COMPRISES"
    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"
    REALIZES = "REALIZES"
    INTERFACES_WITH = "INTERFACES_WITH"
    CONNECTS_TO = "CONNECTS_TO"
    PART_OF = "PART_OF"
    DECOMPOSES_TO = "DECOMPOSES_TO"
    HAS_PART = "HAS_PART"
    INSTANCE_OF = "INSTANCE_OF"
    SUBCLASS_OF = "SUBCLASS_OF"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    GOVERNS = "GOVERNS"
    TRIGGERS = "TRIGGERS"


class OntologyEntityType(str, Enum):
    CONCEPT = "Concept"
    BUSINESS_SCENARIO = "BusinessScenario"
    CAPABILITY = "Capability"
    SYSTEM = "System"
    PROCESS = "Process"
    ACTOR = "Actor"
    DATA = "Data"
    REQUIREMENT = "Requirement"
    ARCHITECTURAL_ELEMENT = "ArchitecturalElement"
    SERVICE = "Service"
    FUNCTION = "Function"
    FEATURE = "Feature"
    CONSTRAINT = "Constraint"
    ASSET = "Asset"
    BOM = "BOM"
    ASSEMBLY = "Assembly"
    MODULE = "Module"
    PART_FAMILY = "Part Family"
    PART = "Part"
    VARIANT = "Variant"
    CONFIGURATION = "Configuration"
    COMPONENT = "Component"


class CanonicalStatus(str, Enum):
    RAW_CANDIDATE = "Raw_Candidate"
    STRUCTURED_CANDIDATE = "Structured_Candidate"
    SEMANTIC_CANDIDATE = "Semantic_Candidate"
    INSTANCE_CANDIDATE = "Instance_Candidate"
    GOVERNED_CANONICAL = "Governed_Canonical"


class EntityType(str, Enum):
    SYSTEM = "System"
    CAPABILITY = "Capability"
    PROCESS = "Process"
    DATA = "Data"
    ASSET = "Asset"
    ACTOR = "Actor"
    RULE = "Rule"
    CONSTRAINT = "Constraint"
    EVENT = "Event"
    INTERFACE = "Interface"


class RelationType(str, Enum):
    CONTAINS = "contains"
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    USES = "uses"
    ASSOCIATED_WITH = "associated_with"
    CALLS = "calls"
    PRODUCES = "produces"
    CONSUMES = "consumes"
    TRIGGERS = "triggers"
    CONFLICTS_WITH = "conflicts_with"


class AttrValueType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    ENUM = "enum"
    OBJECT = "object"
    LIST = "list"


class PipelinePhase(str, Enum):
    P0_RAW = "P0_Raw"
    P1_STRUCTURE = "P1_Structure"
    P2_SEMANTICS = "P2_Semantics"
    P2_5_INSTANCE = "P2_5_Instance"
    P3_GOVERNANCE = "P3_Governance"
    UNKNOWN = "UNKNOWN"