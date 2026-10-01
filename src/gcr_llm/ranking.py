import networkx as nx
import numpy as np

def candidate_scores_from_node_scores(node_scores,candidate_nodes):
    out={}
    for cid,nodes in candidate_nodes.items():
        vals=[node_scores[n] for n in nodes if n in node_scores]
        out[cid]=float(np.mean(vals)) if vals else 0.0
    return out

def rank_graph(graph,candidate_nodes,method="pagerank",damping=0.85,max_iter=100,tolerance=1e-6,communities=None):
    if graph.number_of_nodes()==0:return {c:0.0 for c in candidate_nodes}
    if method=="pagerank": scores=nx.pagerank(graph,alpha=damping,weight="weight",max_iter=max_iter,tol=tolerance)
    elif method=="eigenvector": scores=nx.eigenvector_centrality(graph,max_iter=max_iter,tol=tolerance,weight="weight")
    elif method=="community_size":
        if communities is None: raise ValueError("community_size requires community detection.")
        scores={n:float(len(c)) for c in communities for n in c}
    else: raise ValueError(f"Unsupported graph ranking method: {method}")
    return candidate_scores_from_node_scores(scores,candidate_nodes)

def rank_direct_similarity(graph,candidate_ids):
    return {cid:float(np.mean([d.get("weight",0.0) for _,_,d in graph.edges(i,data=True)])) if graph.degree(i) else 0.0 for i,cid in enumerate(candidate_ids)}

def select_top(scores):
    return max(scores,key=lambda k:(scores[k],k)) if scores else None
