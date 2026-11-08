import os
import yaml
from typing import Dict, Set, Optional, List
from pathlib import Path

class RelationTypeRegistry:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, registry_path: Optional[str] = None):
        if self._initialized:
            return
        
        self._initialized = True
        self._registry_path = registry_path or self._find_registry_path()
        self._relation_types: Dict[str, dict] = {}
        self._categories: Dict[str, dict] = {}
        self._allowed_types: Set[str] = set()
        self._type_aliases: Dict[str, str] = {}
        self._category_types: Dict[str, Set[str]] = {}
        self._load_registry()

    def _find_registry_path(self) -> str:
        config_dir = Path(__file__).resolve().parent.parent.parent.parent / "config" / "ontology"
        registry_path = config_dir / "relation_type_registry.yaml"
        return str(registry_path)

    def _load_registry(self):
        if not os.path.exists(self._registry_path):
            self._load_fallback()
            return

        try:
            with open(self._registry_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            self._categories = data.get('categories', {})
            self._relation_types = data.get('relation_types', {})
            
            for type_id, type_def in self._relation_types.items():
                self._allowed_types.add(type_id)
                
                category = type_def.get('category')
                if category not in self._category_types:
                    self._category_types[category] = set()
                self._category_types[category].add(type_id)
                
                lowercase_id = type_id.lower()
                if lowercase_id != type_id:
                    self._type_aliases[lowercase_id] = type_id
                    self._allowed_types.add(lowercase_id)

        except Exception:
            self._load_fallback()

    def _load_fallback(self):
        self._allowed_types = {
            "SUBSUMES", "DECOMPOSES_TO", "CONTAINS", "HAS_PART", "HAS_VARIANT", "HAS_BOM",
            "REALIZES", "DEPLOYS_TO", "SUPPORTS", "EXECUTES",
            "OWNS", "MANAGES", "MAINTAINS", "ASSIGNED_TO",
            "DEPENDS_ON", "USES", "INTEGRATES_WITH", "REFERENCES",
            "TRACES_TO", "VERIFIES", "VALIDATES", "SATISFIES", "CONFORMS_TO",
            "APPLIES_TO", "EFFECTIVE_FOR", "REPLACES",
            "same_as", "related_to", "child_of", "parent_of", "participates_in",
            "connected_to",
            "governs_data", "results_in", "generates",
            "triggers", "has_step", "modifies",
            "instance_of", "option_for", "derived_from",
            "maps", "feeds", "updates", "populates", "sends_to", "handles_over",
            "maps_to", "processes", "produces", "implements", "deploys",
            "controls", "enhances", "is_part_of", "integratedWith", "monitors", "defines", "FEEDS_INTO",
            "associated_with", "uses", "manages", "has_part", "contains", "depends_on",
            "integrates_with", "deployed_to", "governed_by", "SAME_AS"
        }

        self._type_aliases = {
            "subsumes": "SUBSUMES", "decomposes_to": "DECOMPOSES_TO",
            "contains": "CONTAINS", "has_part": "HAS_PART",
            "has_variant": "HAS_VARIANT", "has_bom": "HAS_BOM",
            "realizes": "REALIZES", "deploys_to": "DEPLOYS_TO",
            "supports": "SUPPORTS", "executes": "EXECUTES",
            "owns": "OWNS", "manages": "MANAGES",
            "maintains": "MAINTAINS", "assigned_to": "ASSIGNED_TO",
            "depends_on": "DEPENDS_ON", "uses": "USES",
            "integrates_with": "INTEGRATES_WITH", "references": "REFERENCES",
            "traces_to": "TRACES_TO", "verifies": "VERIFIES",
            "validates": "VALIDATES", "satisfies": "SATISFIES",
            "conforms_to": "CONFORMS_TO", "applies_to": "APPLIES_TO",
            "effective_for": "EFFECTIVE_FOR", "replaces": "REPLACES",
            "sameas": "same_as", "related_to": "related_to",
            "child_of": "child_of", "parent_of": "parent_of",
            "participates_in": "participates_in", "connected_to": "connected_to",
            "governs_data": "governs_data", "results_in": "results_in",
            "generates": "generates", "triggers": "triggers",
            "has_step": "has_step", "modifies": "modifies",
            "instance_of": "instance_of", "option_for": "option_for",
            "derived_from": "derived_from", "maps": "maps",
            "feeds": "feeds", "updates": "updates", "populates": "populates",
            "sends_to": "sends_to", "handles_over": "handles_over",
            "maps_to": "maps_to", "processes": "processes",
            "produces": "produces", "implements": "implements",
            "deploys": "deploys", "controls": "controls",
            "enhances": "enhances", "is_part_of": "is_part_of",
            "integratedwith": "integratedWith", "monitors": "monitors",
            "defines": "defines", "feeds_into": "FEEDS_INTO",
            "associated_with": "associated_with", "deployed_to": "DEPLOYS_TO",
            "governed_by": "governs_data", "same_as": "same_as"
        }

        self._category_types = {
            "structural": {"SUBSUMES", "DECOMPOSES_TO", "CONTAINS", "HAS_PART", "HAS_VARIANT", "HAS_BOM", "is_part_of"},
            "realization": {"REALIZES", "DEPLOYS_TO", "SUPPORTS", "EXECUTES", "implements", "deploys"},
            "ownership": {"OWNS", "MANAGES", "MAINTAINS", "ASSIGNED_TO"},
            "dependency": {"DEPENDS_ON", "USES", "INTEGRATES_WITH", "REFERENCES", "integratedWith"},
            "traceability": {"TRACES_TO", "VERIFIES", "VALIDATES", "SATISFIES", "CONFORMS_TO"},
            "effectivity": {"APPLIES_TO", "EFFECTIVE_FOR", "REPLACES"},
            "special": {"same_as", "related_to", "child_of", "parent_of", "participates_in", "associated_with"},
            "system": {"connected_to", "controls", "monitors", "defines"},
            "data": {"governs_data", "results_in", "generates", "maps", "maps_to", "updates", "populates", "sends_to", "handles_over"},
            "process": {"triggers", "has_step", "modifies", "feeds", "processes", "produces", "FEEDS_INTO"},
            "asset": {"instance_of", "option_for", "derived_from"},
            "enhancement": {"enhances"},
        }

    def get_allowed_types(self) -> Set[str]:
        return self._allowed_types

    def get_type_aliases(self) -> Dict[str, str]:
        return self._type_aliases

    def get_category_types(self) -> Dict[str, Set[str]]:
        return self._category_types

    def get_relation_type(self, type_id: str) -> Optional[dict]:
        return self._relation_types.get(type_id)

    def is_valid_type(self, type_id: str) -> bool:
        return type_id in self._allowed_types

    def normalize_type(self, type_id: str) -> str:
        if not type_id:
            return "RELATED_TO"
        lower_id = type_id.lower()
        normalized = self._type_aliases.get(lower_id, type_id)
        if normalized not in self._allowed_types:
            return "RELATED_TO"
        return normalized

    def get_types_by_category(self, category: str) -> Set[str]:
        return self._category_types.get(category, set())

    def get_categories(self) -> Dict[str, dict]:
        return self._categories

    def reload(self):
        self._initialized = False
        self.__init__()
