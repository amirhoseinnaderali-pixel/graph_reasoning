import numpy as np
from gcr_llm.graph import candidate_similarity_graph,graph_statistics
from gcr_llm.ranking import rank_direct_similarity
from gcr_llm.schema import Candidate,ReasoningNode

def candidates():
    return [Candidate("a","m1","alpha",trajectory=[ReasoningNode("t1","alpha reasoning"),ReasoningNode("t2","follow alpha",["t1"])]),Candidate("b","m2","alpha alt",trajectory=[ReasoningNode("t1","alpha alternative")]),Candidate("c","m3","beta",trajectory=[ReasoningNode("t1","beta")])]

def test_candidate_graph_has_nodes():
    cs=candidates(); g=candidate_similarity_graph(cs,np.eye(3,dtype=np.float32),threshold=0,top_k_neighbors=2); assert g.number_of_nodes()==3

def test_graph_statistics_schema():
    g=candidate_similarity_graph(candidates(),np.eye(3,dtype=np.float32),threshold=0,top_k_neighbors=2); stats=graph_statistics(g)
    assert {"num_nodes","num_edges","num_similarity_edges","num_dependency_edges","avg_degree","graph_density","num_communities","largest_community","avg_similarity","max_similarity"}.issubset(stats)

def test_direct_ranking_deterministic():
    cs=candidates(); g=candidate_similarity_graph(cs,np.eye(3,dtype=np.float32),threshold=0,top_k_neighbors=2)
    assert rank_direct_similarity(g,[c.candidate_id for c in cs])==rank_direct_similarity(g,[c.candidate_id for c in cs])
