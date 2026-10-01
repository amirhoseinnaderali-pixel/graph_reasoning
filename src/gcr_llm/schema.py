from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

@dataclass
class ReasoningNode:
    node_id: str
    content: str
    dependencies: list[str] = field(default_factory=list)
    thought_type: str | None = None
    confidence: float | None = None

@dataclass
class Candidate:
    candidate_id: str
    model: str
    answer: str
    confidence: float | None = None
    trajectory: list[ReasoningNode] = field(default_factory=list)
    answer_label: str | None = None
    is_correct: bool | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]):
        trajectory=[ReasoningNode(str(x["node_id"]),str(x.get("content","")),[str(d) for d in x.get("dependencies",[])],x.get("thought_type"),float(x["confidence"]) if x.get("confidence") is not None else None) for x in raw.get("trajectory",[])]
        return cls(str(raw["candidate_id"]),str(raw.get("model","unknown")),str(raw.get("answer","")),float(raw["confidence"]) if raw.get("confidence") is not None else None,trajectory,str(raw["answer_label"]) if raw.get("answer_label") is not None else None,raw.get("is_correct"))

def load_problem_records(path: str | Path):
    with open(path,encoding="utf-8") as f: payload=json.load(f)
    if isinstance(payload,dict) and "problems" in payload: records=payload["problems"]
    elif isinstance(payload,list): records=payload
    elif isinstance(payload,dict) and "candidates" in payload: records=[payload]
    else: raise ValueError("JSON must be a problem object, list, or {'problems': [...]} object.")
    out=[]
    for r in records:
        candidates=[Candidate.from_dict(x) for x in r.get("candidates",[])]
        if not candidates: raise ValueError(f"Problem {r.get('problem_id','unknown')} has no candidates.")
        out.append({"problem_id":str(r.get("problem_id","problem-unknown")),"problem":str(r.get("problem","")),"gold_answer":r.get("gold_answer"),"candidates":candidates})
    return out
