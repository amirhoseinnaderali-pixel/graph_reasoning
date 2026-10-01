import hashlib,json,random,time
from pathlib import Path
import numpy as np
from .baselines import single_model,independent_multi_sample,majority_consensus,embedding_similarity_ranking
from .config import ExperimentConfig
from .embeddings import EmbeddingProvider
from .graph import build_trajectory_graph,detect_communities,graph_statistics
from .metrics import evaluate_selection
from .ranking import rank_graph,select_top
from .schema import load_problem_records

def run_problem(record,config,method=None):
    method=method or config.ranking.get("method","pagerank"); candidates=record["candidates"]; started=time.perf_counter()
    random.seed(config.seed); np.random.seed(config.seed)
    if method=="single_model": selected,scores=single_model(candidates); stats={}
    elif method=="independent_multi_sample": selected,scores=independent_multi_sample(candidates); stats={}
    elif method=="majority": selected,scores=majority_consensus(candidates); stats={}
    elif method in ("direct_similarity","pagerank","eigenvector","community_size"):
        provider=EmbeddingProvider(**config.embedding)
        nodes=[t for c in candidates for t in c.trajectory]
        if method=="direct_similarity" or not nodes:
            emb=provider.encode([c.answer for c in candidates])
            selected,scores=embedding_similarity_ranking(candidates,emb,float(config.graph.get("similarity_threshold",.5)),int(config.graph.get("top_k_neighbors",5)))
            stats={} if method=="direct_similarity" else {"warning":"No trajectory supplied; answer-level similarity fallback used."}
        else:
            emb=provider.encode([n.content for n in nodes])
            graph,candidate_nodes=build_trajectory_graph(candidates,emb,float(config.graph.get("similarity_threshold",.5)),int(config.graph.get("top_k_neighbors",5)),bool(config.graph.get("include_similarity_edges",True)),bool(config.graph.get("include_dependency_edges",True)),bool(config.graph.get("cross_model_similarity_only",True)),float(config.graph.get("dependency_weight",1)))
            communities=detect_communities(graph,float(config.community.get("resolution",1)),config.seed) if config.community.get("enabled",True) else []
            scores=rank_graph(graph,candidate_nodes,method,float(config.ranking.get("damping",.85)),int(config.ranking.get("max_iter",100)),float(config.ranking.get("tolerance",1e-6)),communities)
            selected=select_top(scores); stats=graph_statistics(graph,communities); stats["candidate_node_counts"]={k:len(v) for k,v in candidate_nodes.items()}
    else: raise ValueError(f"Unknown method: {method}")
    elapsed=time.perf_counter()-started; correct=evaluate_selection(candidates,selected,record.get("gold_answer"))
    eid=hashlib.sha1(f"{record['problem_id']}|{method}|{config.seed}|{config.graph}".encode()).hexdigest()[:12]
    return {"experiment_id":eid,"problem_id":record["problem_id"],"method":method,"embedding_model":config.embedding.get("model_name"),"embedding_backend":config.embedding.get("backend"),"num_models":len({c.model for c in candidates}),"num_candidates":len(candidates),"similarity_threshold":config.graph.get("similarity_threshold"),"ranking_method":method,"selected_candidate":selected,"candidate_scores":scores,"correct":correct,"latency_seconds":elapsed,"graph":stats,"seed":config.seed,"status":"evaluated" if correct is not None else "not_evaluated"}

def run_dataset(input_path,output_path,config,method=None):
    records=load_problem_records(input_path); results=[run_problem(r,config,method) for r in records]; path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8") as f: json.dump(results,f,indent=2,ensure_ascii=False)
    return results
