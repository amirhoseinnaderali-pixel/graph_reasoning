from collections import defaultdict
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def build_trajectory_graph(candidates, node_embeddings, similarity_threshold=0.5, top_k_neighbors=5, include_similarity_edges=True, include_dependency_edges=True, cross_model_similarity_only=True, dependency_weight=1.0):
    graph=nx.Graph(); index={}; candidate_nodes=defaultdict(list); i=0
    for c in candidates:
        for t in c.trajectory:
            graph.add_node(i,node_id=t.node_id,candidate_id=c.candidate_id,model=c.model,content=t.content,thought_type=t.thought_type or "unspecified",confidence=t.confidence)
            index[f"{c.candidate_id}:{t.node_id}"]=i; candidate_nodes[c.candidate_id].append(i); i+=1
    if include_dependency_edges:
        for c in candidates:
            for t in c.trajectory:
                dst=index.get(f"{c.candidate_id}:{t.node_id}")
                for dep in t.dependencies:
                    src=index.get(f"{c.candidate_id}:{dep}")
                    if src is not None and dst is not None:
                        graph.add_edge(src,dst,weight=float(dependency_weight),edge_type="dependency")
    if include_similarity_edges and graph.number_of_nodes()>1:
        if len(node_embeddings)!=graph.number_of_nodes(): raise ValueError("Embedding count must match reasoning nodes.")
        sim=cosine_similarity(node_embeddings); ids=list(graph.nodes())
        for row,src in enumerate(ids):
            vals=sim[row].copy(); vals[row]=-1
            for col in np.argsort(vals)[::-1][:min(top_k_neighbors,len(ids)-1)]:
                weight=float(sim[row,col]); dst=ids[col]
                if weight<similarity_threshold: continue
                if cross_model_similarity_only and graph.nodes[src]["model"]==graph.nodes[dst]["model"]: continue
                if graph.has_edge(src,dst):
                    graph[src][dst].update(edge_type="hybrid",similarity=weight,weight=max(weight,float(graph[src][dst].get("weight",0))))
                else: graph.add_edge(src,dst,weight=weight,edge_type="similarity")
    return graph,dict(candidate_nodes)

def candidate_similarity_graph(candidates, embeddings, threshold=0.5, top_k_neighbors=5, cross_model_only=False):
    g=nx.Graph()
    for i,c in enumerate(candidates): g.add_node(i,candidate_id=c.candidate_id,model=c.model)
    if len(candidates)<2:return g
    sim=cosine_similarity(embeddings)
    for i in range(len(candidates)):
        vals=sim[i].copy(); vals[i]=-1
        for j in np.argsort(vals)[::-1][:min(top_k_neighbors,len(candidates)-1)]:
            if sim[i,j]>=threshold and not (cross_model_only and candidates[i].model==candidates[j].model):
                g.add_edge(i,int(j),weight=float(sim[i,j]))
    return g

def graph_statistics(graph, communities=None):
    edge_types=defaultdict(int); weights=[]; deps=0
    for _,_,d in graph.edges(data=True):
        typ=d.get("edge_type","unknown"); edge_types[typ]+=1
        if typ in ("similarity","hybrid"): weights.append(float(d.get("similarity",d.get("weight",0))))
        if typ in ("dependency","hybrid"): deps+=1
    n=graph.number_of_nodes(); degrees=[d for _,d in graph.degree()]; sizes=[len(c) for c in communities or []]
    return {"num_nodes":n,"num_edges":graph.number_of_edges(),"num_similarity_edges":edge_types["similarity"]+edge_types["hybrid"],"num_dependency_edges":deps,"avg_degree":float(np.mean(degrees)) if degrees else 0.0,"graph_density":float(nx.density(graph)) if n>1 else 0.0,"num_communities":len(sizes),"largest_community":max(sizes) if sizes else 0,"avg_similarity":float(np.mean(weights)) if weights else 0.0,"max_similarity":float(np.max(weights)) if weights else 0.0,"connected_components":nx.number_connected_components(graph) if n else 0}

def detect_communities(graph,resolution=1.0,seed=42):
    return [set(c) for c in nx.community.louvain_communities(graph,resolution=resolution,seed=seed)] if graph.number_of_nodes() else []
