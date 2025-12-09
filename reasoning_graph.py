

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
    USE_TRANSFORMER = True
    print("✓ Using transformer embeddings")
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    EMBEDDING_MODEL = TfidfVectorizer(max_features=1024)
    USE_TRANSFORMER = False
    print("⚠ Using TF-IDF embeddings")


@dataclass
class GraphConfig:
    """Configuration for graph construction"""
    chunk_size: int = 300  # words per chunk
    min_chunk_words: int = 20  # minimum words for valid chunk
    similarity_threshold: float = 0.5  # minimum similarity for edge
    top_k_neighbors: int = 5  # number of similar chunks to connect
    community_resolution: float = 1.0  # for community detection


class ImprovedChainOfThoughtGraph:
    """
    Improved graph representation with:
    - No redundant merge/connection nodes
    - Direct edges with similarity weights
    - Sparse similarity connections
    - Community detection
    """
    
    COLORS = {
        'problem': '#FF6B6B',
        'solution': '#4ECDC4', 
        'chunk': '#45B7D1',
        'community': '#FFE66D'
    }
    
    def __init__(self, config: GraphConfig = None):
        # ✅ UNDIRECTED for symmetric relationships
        self.graph = nx.Graph()
        self.config = config or GraphConfig()
        self.node_counter = 0
        
        # Data structures
        self.chunks: Dict[int, dict] = {}
        self.solutions: Dict = {}
        self.communities: Dict[int, Set[int]] = {}
        
    def add_problem_node(self, problem_text: str) -> int:
        """Add root problem node"""
        node_id = self._next_id()
        self.graph.add_node(
            node_id,
            label="PROBLEM",
            text=problem_text[:200],  # truncate for readability
            type='problem',
            level=0
        )
        return node_id
    
    def add_solution_with_chunks(
        self, 
        solution_data: dict,
        solution_id: int,
        parent_node: int
    ) -> int:
        """Create solution node with chunks"""
        # Solution root
        solution_root = self._next_id()
        solution_name = solution_data.get('name', f'Solution {solution_id}')
        
        self.graph.add_node(
            solution_root,
            label=f"SOL_{solution_id}",
            text=solution_name,
            type='solution',
            level=1,
            solution_id=solution_id
        )
        self.graph.add_edge(parent_node, solution_root)
        
        # Store metadata
        self.solutions[solution_id] = {
            'root': solution_root,
            'chunks': [],
            'name': solution_name
        }
        
        # Create chunks
        full_text = self._combine_solution_text(solution_data)
        chunks = self._split_text(full_text)
        
        print(f"  Solution {solution_id}: {len(chunks)} chunks")
        
        for idx, chunk_text in enumerate(chunks):
            chunk_id = self._create_chunk_node(
                chunk_text, solution_root, solution_id, idx
            )
            self.solutions[solution_id]['chunks'].append(chunk_id)
        
        return solution_root
    
    def add_strategy_with_chunks(
        self,
        strategy_data: dict,
        parent_node: int
    ) -> int:
        """Add strategy as chunked nodes"""
        strategy_root = self._next_id()
        self.graph.add_node(
            strategy_root,
            label="STRATEGY",
            text="Detailed Strategy",
            type='solution',
            level=1
        )
        self.graph.add_edge(parent_node, strategy_root)
        
        self.solutions['strategy'] = {
            'root': strategy_root,
            'chunks': [],
            'name': 'Strategy'
        }
        
        full_text = self._combine_strategy_text(strategy_data)
        chunks = self._split_text(full_text)
        
        print(f"  Strategy: {len(chunks)} chunks")
        
        for idx, chunk_text in enumerate(chunks):
            chunk_id = self._create_chunk_node(
                chunk_text, strategy_root, 'strategy', idx
            )
            self.solutions['strategy']['chunks'].append(chunk_id)
        
        return strategy_root
    
    def compute_embeddings(self):
        """Compute embeddings (TF-IDF only)"""
        if not USE_TRANSFORMER:
            print("\n⚙ Computing TF-IDF embeddings...")
            texts = [c['text'] for c in self.chunks.values()]
            embeddings = EMBEDDING_MODEL.fit_transform(texts).toarray()
            
            for idx, (nid, chunk) in enumerate(self.chunks.items()):
                chunk['embedding'] = embeddings[idx]
    
    def connect_similar_chunks(self):
        """
        ✅ IMPROVED: Direct edges between similar chunks
        - Uses sparse k-NN for scalability
        - Only connects top-k most similar
        - Edges have similarity weights
        """
        print(f"\n🔍 Finding similar chunks (top-{self.config.top_k_neighbors})...")
        
        # Get embeddings
        node_ids = list(self.chunks.keys())
        embeddings = np.array([self.chunks[nid]['embedding'] for nid in node_ids])
        
        # ✅ Use k-NN for sparse connections
        n_neighbors = min(self.config.top_k_neighbors + 1, len(node_ids))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        nn.fit(embeddings)
        
        distances, indices = nn.kneighbors(embeddings)
        
        edge_count = 0
        for i, node_i in enumerate(node_ids):
            sol_i = self.chunks[node_i]['solution_id']
            
            for j, dist in zip(indices[i][1:], distances[i][1:]):
                node_j = node_ids[j]
                sol_j = self.chunks[node_j]['solution_id']
                
                # Only connect different solutions
                if sol_i == sol_j:
                    continue
                
                similarity = 1 - dist  # cosine distance to similarity
                
                if similarity >= self.config.similarity_threshold:
                    # ✅ Direct edge with weight
                    if not self.graph.has_edge(node_i, node_j):
                        self.graph.add_edge(
                            node_i, node_j,
                            weight=similarity,
                            edge_type='similarity'
                        )
                        edge_count += 1
        
        print(f"  ✓ Created {edge_count} similarity edges")
        return edge_count
    
    def detect_communities(self):
        """
        ✅ NEW: Community detection on chunk subgraph
        Finds clusters of related chunks across solutions
        """
        print("\n🔍 Detecting communities...")
        
        # Extract chunk subgraph
        chunk_nodes = [n for n in self.graph.nodes() 
                      if self.graph.nodes[n].get('type') == 'chunk']
        subgraph = self.graph.subgraph(chunk_nodes)
        
        # Louvain community detection
        communities = nx.community.louvain_communities(
            subgraph,
            resolution=self.config.community_resolution,
            seed=42
        )
        
        # Store communities
        for comm_id, nodes in enumerate(communities):
            self.communities[comm_id] = nodes
            for node in nodes:
                self.graph.nodes[node]['community'] = comm_id
        
        print(f"  ✓ Found {len(communities)} communities")
        return communities
    
    def visualize(self, layout: str = 'spring', figsize=(30, 20)):
        """Enhanced visualization with communities"""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Layout
        if layout == 'spring':
            pos = nx.spring_layout(self.graph, k=3, iterations=50, seed=42)
        else:
            pos = nx.kamada_kawai_layout(self.graph)
        
        # Node properties
        colors, sizes = self._get_node_properties()
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=colors,
            node_size=sizes,
            alpha=0.8,
            ax=ax
        )
        
        # Draw edges by type
        self._draw_edges_by_type(pos, ax)
        
        # Labels
        labels = {n: self.graph.nodes[n].get('label', str(n))[:12] 
                 for n in self.graph.nodes()}
        nx.draw_networkx_labels(
            self.graph, pos, labels,
            font_size=7, font_weight='bold', ax=ax
        )
        
        # Title and legend
        self._add_visualization_info(ax)
        
        plt.tight_layout()
        return fig
    
    def get_statistics(self) -> dict:
        """Comprehensive graph statistics"""
        chunk_nodes = [n for n in self.graph.nodes() 
                      if self.graph.nodes[n].get('type') == 'chunk']
        
        # Edge statistics
        sim_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True)
                    if d.get('edge_type') == 'similarity']
        
        if sim_edges:
            weights = [d['weight'] for _, _, d in sim_edges]
            avg_sim = np.mean(weights)
            max_sim = np.max(weights)
        else:
            avg_sim = max_sim = 0
        
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'num_solutions': len(self.solutions),
            'total_chunks': len(self.chunks),
            'num_communities': len(self.communities),
            'similarity_edges': len(sim_edges),
            'avg_similarity': float(avg_sim),
            'max_similarity': float(max_sim),
            'avg_degree': np.mean([d for _, d in self.graph.degree(chunk_nodes)]),
            'density': nx.density(self.graph.subgraph(chunk_nodes)),
            'config': {
                'chunk_size': self.config.chunk_size,
                'similarity_threshold': self.config.similarity_threshold,
                'top_k': self.config.top_k_neighbors
            }
        }
        
        # Per-solution stats
        stats['chunks_per_solution'] = {
            f'Solution_{sid}': len(sdata['chunks'])
            for sid, sdata in self.solutions.items()
        }
        
        # Community stats
        if self.communities:
            stats['community_sizes'] = {
                f'Community_{cid}': len(nodes)
                for cid, nodes in self.communities.items()
            }
        
        return stats
    
    def export_data(
        self,
        graph_file: str = 'graph.gexf',
        json_file: str = 'data.json',
        similarity_file: str = 'similarity.npy'
    ):
        """Export all data"""
        # Graph
        clean_graph = self._create_clean_graph()
        nx.write_gexf(clean_graph, graph_file)
        print(f"✓ Graph: {graph_file}")
        
        # Similarity matrix (full, for analysis)
        if len(self.chunks) > 0:
            node_ids = list(self.chunks.keys())
            embeddings = np.array([self.chunks[nid]['embedding'] for nid in node_ids])
            sim_matrix = cosine_similarity(embeddings)
            np.save(similarity_file, sim_matrix)
            print(f"✓ Similarity matrix: {similarity_file}")
        
        # JSON metadata
        self._export_json(json_file)
        print(f"✓ Metadata: {json_file}")
    
    def print_analysis(self):
        """Print detailed analysis"""
        print("\n" + "=" * 80)
        print("GRAPH ANALYSIS")
        print("=" * 80)
        
        stats = self.get_statistics()
        
        print(f"\n📊 OVERVIEW:")
        print(f"  Nodes: {stats['total_nodes']}")
        print(f"  Edges: {stats['total_edges']}")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Solutions: {stats['num_solutions']}")
        print(f"  Communities: {stats['num_communities']}")
        
        print(f"\n🔗 SIMILARITY:")
        print(f"  Similarity edges: {stats['similarity_edges']}")
        print(f"  Avg similarity: {stats['avg_similarity']:.3f}")
        print(f"  Max similarity: {stats['max_similarity']:.3f}")
        print(f"  Avg degree: {stats['avg_degree']:.2f}")
        print(f"  Graph density: {stats['density']:.4f}")
        
        if self.communities:
            print(f"\n🏘️ COMMUNITIES:")
            for cid, nodes in sorted(self.communities.items()):
                sol_dist = defaultdict(int)
                for node in nodes:
                    sol_id = self.chunks[node]['solution_id']
                    sol_dist[sol_id] += 1
                
                print(f"  Community {cid}: {len(nodes)} chunks")
                # Sort by solution_id (handle mixed int/str types by converting all to string)
                for sol_id, count in sorted(sol_dist.items(), key=lambda x: (str(x[0]), x[1])):
                    print(f"    - Solution {sol_id}: {count} chunks")
    
    # ==================== PRIVATE METHODS ====================
    
    def _next_id(self) -> int:
        nid = self.node_counter
        self.node_counter += 1
        return nid
    
    def _split_text(self, text: str) -> List[str]:
        """Split into chunks"""
        text = ' '.join(text.split())
        words = text.split()
        
        chunks = []
        for i in range(0, len(words), self.config.chunk_size):
            chunk = ' '.join(words[i:i + self.config.chunk_size])
            if len(chunk.split()) >= self.config.min_chunk_words:
                chunks.append(chunk)
        
        return chunks
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding"""
        if USE_TRANSFORMER:
            return EMBEDDING_MODEL.encode(text, convert_to_numpy=True)
        return None
    
    def _combine_solution_text(self, sol: dict) -> str:
        """Combine solution text"""
        return " ".join([
            sol.get('core_idea', ''),
            sol.get('novelty', ''),
            sol.get('required_assumptions', ''),
            sol.get('mathematical_tools', '')
        ])
    
    def _combine_strategy_text(self, strat: dict) -> str:
        """Combine strategy text"""
        text = strat.get('overview', '')
        for step in strat.get('key_steps', []):
            text += " " + step.get('description', '')
            text += " " + step.get('justification', '')
            text += " " + step.get('potential_techniques', '')
        return text
    
    def _create_chunk_node(
        self,
        chunk_text: str,
        parent: int,
        solution_id,
        chunk_idx: int
    ) -> int:
        """Create chunk node"""
        node_id = self._next_id()
        embedding = self._get_embedding(chunk_text)
        
        self.graph.add_node(
            node_id,
            label=f"S{solution_id}_C{chunk_idx+1}",
            text=chunk_text[:100],  # truncate for graph
            type='chunk',
            level=2,
            solution_id=solution_id,
            chunk_id=chunk_idx
        )
        self.graph.add_edge(parent, node_id)
        
        self.chunks[node_id] = {
            'text': chunk_text,  # full text
            'embedding': embedding,
            'solution_id': solution_id,
            'chunk_id': chunk_idx,
            'node_id': node_id
        }
        
        return node_id
    
    def _get_node_properties(self) -> Tuple[List, List]:
        """Get colors and sizes"""
        colors = []
        sizes = []
        
        size_map = {
            'problem': 5000,
            'solution': 3000,
            'chunk': 800
        }
        
        for node in self.graph.nodes():
            ntype = self.graph.nodes[node].get('type', 'chunk')
            
            # Color by community if available
            if ntype == 'chunk' and 'community' in self.graph.nodes[node]:
                comm = self.graph.nodes[node]['community']
                # Color palette for communities
                palette = plt.cm.Set3(np.linspace(0, 1, 12))
                colors.append(palette[comm % 12])
            else:
                colors.append(self.COLORS.get(ntype, '#CCCCCC'))
            
            sizes.append(size_map.get(ntype, 800))
        
        return colors, sizes
    
    def _draw_edges_by_type(self, pos, ax):
        """Draw edges with different styles"""
        # Hierarchy edges (thin gray)
        hier_edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                     if d.get('edge_type') != 'similarity']
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=hier_edges,
            edge_color='#CCCCCC', alpha=0.3, width=1, ax=ax
        )
        
        # Similarity edges (colored by weight)
        sim_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True)
                    if d.get('edge_type') == 'similarity']
        
        if sim_edges:
            weights = [d['weight'] for _, _, d in sim_edges]
            edges_only = [(u, v) for u, v, _ in sim_edges]
            
            nx.draw_networkx_edges(
                self.graph, pos, edgelist=edges_only,
                edge_color=weights,
                edge_cmap=plt.cm.RdYlGn,
                edge_vmin=0.5, edge_vmax=1.0,
                alpha=0.6, width=2, ax=ax
            )
    
    def _add_visualization_info(self, ax):
        """Add title and legend"""
        from matplotlib.patches import Patch
        
        stats = self.get_statistics()
        
        ax.set_title(
            f"Chain of Thought Graph\n"
            f"Chunks: {stats['total_chunks']} | "
            f"Similarities: {stats['similarity_edges']} | "
            f"Communities: {stats['num_communities']}",
            fontsize=18, fontweight='bold', pad=20
        )
        ax.axis('off')
        
        legend = [
            Patch(facecolor=c, label=t.upper())
            for t, c in self.COLORS.items()
        ]
        ax.legend(handles=legend, loc='upper left', fontsize=10)
    
    def _create_clean_graph(self) -> nx.Graph:
        """Create serializable graph"""
        clean = nx.Graph()
        
        for node, data in self.graph.nodes(data=True):
            clean_data = {
                k: (float(v) if isinstance(v, np.floating) else
                    int(v) if isinstance(v, np.integer) else
                    str(v) if isinstance(v, (list, np.ndarray)) else v)
                for k, v in data.items()
            }
            clean.add_node(node, **clean_data)
        
        for u, v, data in self.graph.edges(data=True):
            clean_data = {
                k: (float(v) if isinstance(v, np.floating) else
                    int(v) if isinstance(v, np.integer) else v)
                for k, v in data.items()
            }
            clean.add_edge(u, v, **clean_data)
        
        return clean
    
    def _export_json(self, json_file: str):
        """Export metadata"""
        data = {
            'metadata': self.get_statistics(),
            'chunks': [
                {
                    'node_id': int(nid),
                    'solution_id': c['solution_id'],
                    'chunk_id': int(c['chunk_id']),
                    'text': c['text'],
                    'community': self.graph.nodes[nid].get('community', -1)
                }
                for nid, c in self.chunks.items()
            ],
            'solutions': {
                str(sid): {
                    'name': s['name'],
                    'num_chunks': len(s['chunks'])
                }
                for sid, s in self.solutions.items()
            }
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ==================== DATA LOADING ====================

def load_problem_data(data_dir: str = 'data') -> dict:
    """Load all JSON files"""
    data_path = Path(data_dir)
    
    problem_data = None
    all_approaches = []
    
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    for i in range(1, 6):  # try multiple files
        file_path = data_path / f'creasoning_{i}.json'
        
        if not file_path.exists():
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        if isinstance(file_data, dict):
            file_data = [file_data]
        
        print(f"\n📁 {file_path.name}: {len(file_data)} model(s)")
        
        for model_data in file_data:
            if problem_data is None:
                problem_data = {
                    'problem_restatement': model_data.get('problem_restatement', ''),
                    'detailed_strategy': model_data.get('detailed_strategy', {})
                }
            
            approaches = model_data.get('proposed_approaches', [])
            all_approaches.extend(approaches)
    
    if problem_data is None:
        raise ValueError("No data files found!")
    
    problem_data['proposed_approaches'] = all_approaches
    
    print(f"\n✓ Loaded {len(all_approaches)} approaches")
    print("=" * 80)
    
    return problem_data


# ==================== MAIN ====================

def reasoning_graph():
    """Main execution"""
    print("=" * 80)
    print("IMPROVED CHAIN OF THOUGHT GRAPH")
    print("=" * 80)
    
    # Config
    config = GraphConfig(
        chunk_size=300,
        min_chunk_words=20,
        similarity_threshold=0.5,
        top_k_neighbors=5,
        community_resolution=1.0
    )
    
    # Load data
    problem_data = load_problem_data('data')
    
    # Build graph
    graph = ImprovedChainOfThoughtGraph(config)
    
    problem_node = graph.add_problem_node(
        problem_data['problem_restatement']
    )
    print(f"\n✓ Problem node: {problem_node}")
    
    print("\n" + "-" * 80)
    print("CREATING CHUNKS")
    print("-" * 80)
    
    for approach in problem_data['proposed_approaches']:
        graph.add_solution_with_chunks(
            approach,
            approach['approach_number'],
            problem_node
        )
    
    graph.add_strategy_with_chunks(
        problem_data['detailed_strategy'],
        problem_node
    )
    
    # Embeddings
    if not USE_TRANSFORMER:
        graph.compute_embeddings()
    
    # Connect similar chunks
    print("\n" + "-" * 80)
    print("FINDING SIMILARITIES")
    print("-" * 80)
    graph.connect_similar_chunks()
    
    # Communities
    print("\n" + "-" * 80)
    print("DETECTING COMMUNITIES")
    print("-" * 80)
    graph.detect_communities()
    
    # Analysis
    graph.print_analysis()
    
    # Export
    print("\n" + "=" * 80)
    print("EXPORTING")
    print("=" * 80)
    graph.export_data(
        'improved_graph.gexf',
        'improved_data.json',
        'improved_similarity.npy'
    )
    
    # Visualize
    print("\n" + "=" * 80)
    print("VISUALIZING")
    print("=" * 80)
    fig = graph.visualize(layout='spring', figsize=(30, 20))
    plt.savefig('improved_graph.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: improved_graph.png")
    
    print("\n" + "=" * 80)
    print("✅ DONE!")
    print("=" * 80)


