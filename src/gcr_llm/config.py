from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import copy
import yaml

@dataclass
class ExperimentConfig:
    seed: int = 42
    embedding: dict[str, Any] = field(default_factory=lambda: {"backend":"tfidf","model_name":"mixedbread-ai/mxbai-embed-large-v1","max_features":2048})
    graph: dict[str, Any] = field(default_factory=lambda: {"similarity_threshold":0.5,"top_k_neighbors":5,"include_similarity_edges":True,"include_dependency_edges":True,"cross_model_similarity_only":True,"dependency_weight":1.0})
    community: dict[str, Any] = field(default_factory=lambda: {"enabled":True,"algorithm":"louvain","resolution":1.0})
    ranking: dict[str, Any] = field(default_factory=lambda: {"method":"pagerank","damping":0.85,"max_iter":100,"tolerance":1e-6})
    experiment: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path):
        with open(path, encoding="utf-8") as f: raw = yaml.safe_load(f) or {}
        return cls(seed=int(raw.get("seed",42)), embedding=dict(raw.get("embedding",{})), graph=dict(raw.get("graph",{})), community=dict(raw.get("community",{})), ranking=dict(raw.get("ranking",{})), experiment=dict(raw.get("experiment",{})))

    def merged(self, override: dict[str, Any]):
        data = {k: copy.deepcopy(getattr(self,k)) for k in ("embedding","graph","community","ranking","experiment")}
        data["seed"] = self.seed
        _deep_update(data, override)
        return ExperimentConfig(**data)

def _deep_update(target, source):
    for k,v in source.items():
        if isinstance(v,dict) and isinstance(target.get(k),dict): _deep_update(target[k],v)
        else: target[k]=v
