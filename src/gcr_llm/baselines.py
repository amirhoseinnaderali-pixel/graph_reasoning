from collections import Counter
import re
import numpy as np
from .graph import candidate_similarity_graph
from .ranking import rank_direct_similarity,select_top

def single_model(candidates):
    if not candidates:return "",{}
    chosen=next(c for c in candidates if c.model==candidates[0].model)
    return chosen.candidate_id,{c.candidate_id:float(c is chosen) for c in candidates}

def independent_multi_sample(candidates):
    scores={c.candidate_id:float(c.confidence or 0.0) for c in candidates}
    if not any(scores.values()):return single_model(candidates)
    return select_top(scores),scores

def embedding_similarity_ranking(candidates,embeddings,threshold,top_k):
    graph=candidate_similarity_graph(candidates,embeddings,threshold,top_k)
    scores=rank_direct_similarity(graph,[c.candidate_id for c in candidates])
    return select_top(scores),scores

def majority_consensus(candidates):
    labels=[(c.candidate_id,c.answer_label if c.answer_label is not None else _norm(c.answer)) for c in candidates]
    counts=Counter(label for _,label in labels); winner=max(counts,key=lambda x:(counts[x],x))
    return next(cid for cid,label in labels if label==winner),{cid:float(counts[label]) for cid,label in labels}

def _norm(text):
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9 ]+"," ",text.lower())).strip()
