
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Embeddings - استفاده از SentenceTransformer برای embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
    USE_TRANSFORMER = True
    print("✓ Using transformer embeddings: mixedbread-ai/mxbai-embed-large-v1")
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    EMBEDDING_MODEL = None
    USE_TRANSFORMER = False
    print("⚠ Using TF-IDF embeddings (will initialize later)")


@dataclass
class GoTConfig:
    """Configuration for Graph of Thoughts"""
    similarity_threshold: float = 0.6  # minimum similarity for cross-solution edges
    top_k_neighbors: int = 3  # top-k similar thoughts to connect
    community_resolution: float = 1.2  # for community detection
    

class GraphOfThoughts:
    """
    Graph of Thoughts: Merges multiple Chain of Thought trees into unified graph
    
    Node Types:
    - problem: root problem node
    - solution: solution root (per model/approach)
    - thought: individual thought from chain_of_thought
    
    Edge Types:
    - hierarchy: parent-child in tree structure
    - dependency: thought dependencies within same solution
    - similarity: cross-solution thought similarities
    """
    
    THOUGHT_TYPE_COLORS = {
        'problem_analysis': '#FF6B6B',
        'hypothesis': '#4ECDC4',
        'exploration': '#45B7D1',
        'obstacle_identification': '#FFA07A',
        'technique_selection': '#98D8C8',
        'synthesis': '#FFE66D',
        'conclusion': '#B19CD9',
        'default': '#95A5A6'
    }
    
    def __init__(self, config: GoTConfig = None):
        self.graph = nx.DiGraph()  # Directed for dependencies
        self.config = config or GoTConfig()
        self.node_counter = 0
        
        # Data structures
        self.thoughts: Dict[int, dict] = {}  # node_id -> thought data
        self.solutions: Dict[str, dict] = {}  # solution_key -> metadata
        self.communities: Dict[int, Set[int]] = {}
        
    def add_problem_node(self, problem_text: str) -> int:
        """Add root problem node"""
        node_id = self._next_id()
        self.graph.add_node(
            node_id,
            label="PROBLEM",
            text=problem_text[:200],
            type='problem',
            level=0
        )
        return node_id
    
    def add_solution_with_cot(
        self, 
        solution_data: dict,
        model_name: str,
        solution_id: int,
        parent_node: int
    ) -> int:
        """
        Add solution with its chain of thought as tree
        
        Args:
            solution_data: dict containing 'chain_of_thought' list
            model_name: name of the model (e.g., 'gemini-2.0-flash')
            solution_id: unique solution identifier
            parent_node: problem node to attach to
        """
        # Solution root
        solution_root = self._next_id()
        solution_key = f"{model_name}_{solution_id}"
        solution_name = solution_data.get('name', f'Solution {solution_id}')
        
        self.graph.add_node(
            solution_root,
            label=f"{model_name[:8]}_S{solution_id}",
            text=solution_name,
            type='solution',
            level=1,
            solution_key=solution_key,
            model=model_name
        )
        self.graph.add_edge(parent_node, solution_root, edge_type='hierarchy')
        
        # Store metadata
        self.solutions[solution_key] = {
            'root': solution_root,
            'thoughts': [],
            'name': solution_name,
            'model': model_name,
            'solution_id': solution_id
        }
        
        # Process chain of thought
        cot = solution_data.get('chain_of_thought', [])
        print(f"  {solution_key}: {len(cot)} thoughts")
        
        if not cot:
            return solution_root
        
        # Create thought nodes and build dependency tree
        thought_id_to_node = {}  # thought_id (e.g., "T1") -> node_id
        
        for thought in cot:
            node_id = self._create_thought_node(
                thought, solution_root, solution_key
            )
            thought_id_to_node[thought['thought_id']] = node_id
            self.solutions[solution_key]['thoughts'].append(node_id)
        
        # Add dependency edges
        for thought in cot:
            source_node = thought_id_to_node[thought['thought_id']]
            for dep_id in thought.get('dependencies', []):
                if dep_id in thought_id_to_node:
                    target_node = thought_id_to_node[dep_id]
                    self.graph.add_edge(
                        target_node, source_node, 
                        edge_type='dependency'
                    )
        
        return solution_root
    
    def compute_embeddings(self):
        """Compute embeddings for all thoughts"""
        global EMBEDDING_MODEL
        if USE_TRANSFORMER:
            print("\n⚙ Computing transformer embeddings...")
            texts = [t['content'] for t in self.thoughts.values()]
            embeddings = EMBEDDING_MODEL.encode(texts, convert_to_numpy=True)
            
            for idx, (node_id, thought) in enumerate(self.thoughts.items()):
                thought['embedding'] = embeddings[idx]
        else:
            print("\n⚙ Computing TF-IDF embeddings...")
            from sklearn.feature_extraction.text import TfidfVectorizer
            texts = [t['content'] for t in self.thoughts.values()]
            EMBEDDING_MODEL = TfidfVectorizer(max_features=512)
            embeddings = EMBEDDING_MODEL.fit_transform(texts).toarray()
            
            for idx, (node_id, thought) in enumerate(self.thoughts.items()):
                thought['embedding'] = embeddings[idx]
    
    def connect_similar_thoughts(self):
        """
        Connect similar thoughts across different solutions
        Creates undirected similarity edges
        """
        print(f"\n🔍 Finding similar thoughts (threshold={self.config.similarity_threshold})...")
        
        if not self.thoughts:
            print("  ⚠ No thoughts to connect")
            return 0
        
        # Get embeddings
        node_ids = list(self.thoughts.keys())
        embeddings = np.array([self.thoughts[nid]['embedding'] for nid in node_ids])
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings)
        
        edge_count = 0
        for i, node_i in enumerate(node_ids):
            sol_i = self.thoughts[node_i]['solution_key']
            
            # Get top-k similar thoughts
            similarities = sim_matrix[i]
            top_k_idx = np.argsort(similarities)[::-1][1:self.config.top_k_neighbors+1]
            
            for j in top_k_idx:
                if similarities[j] < self.config.similarity_threshold:
                    continue
                
                node_j = node_ids[j]
                sol_j = self.thoughts[node_j]['solution_key']
                
                # Only connect different solutions
                if sol_i == sol_j:
                    continue
                
                # Add undirected similarity edge
                if not self.graph.has_edge(node_i, node_j):
                    self.graph.add_edge(
                        node_i, node_j,
                        weight=float(similarities[j]),
                        edge_type='similarity'
                    )
                    self.graph.add_edge(
                        node_j, node_i,
                        weight=float(similarities[j]),
                        edge_type='similarity'
                    )
                    edge_count += 1
        
        print(f"  ✓ Created {edge_count} similarity edges")
        return edge_count
    
    def detect_communities(self):
        """
        Community detection on thought subgraph
        Uses undirected version for community detection
        """
        print("\n🔍 Detecting communities...")
        
        # Extract thought nodes
        thought_nodes = [n for n in self.graph.nodes() 
                        if self.graph.nodes[n].get('type') == 'thought']
        
        if not thought_nodes:
            print("  ⚠ No thoughts for community detection")
            return []
        
        # Create undirected subgraph (ignore edge direction for communities)
        subgraph = self.graph.subgraph(thought_nodes).to_undirected()
        
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
    
    def visualize(self, layout: str = 'hierarchical', figsize=(40, 30)):
        """
        Visualize Graph of Thoughts
        
        Layouts:
        - hierarchical: tree-like (best for seeing solution structure)
        - spring: force-directed (best for seeing similarities)
        - kamada: balanced
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Choose layout
        if layout == 'hierarchical':
            pos = self._hierarchical_layout()
        elif layout == 'spring':
            pos = nx.spring_layout(self.graph, k=5, iterations=100, seed=42)
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
        labels = self._get_node_labels()
        nx.draw_networkx_labels(
            self.graph, pos, labels,
            font_size=6, font_weight='bold', ax=ax
        )
        
        # Title and legend
        self._add_visualization_info(ax)
        
        plt.tight_layout()
        return fig
    
    def get_statistics(self) -> dict:
        """Comprehensive graph statistics"""
        thought_nodes = [n for n in self.graph.nodes() 
                        if self.graph.nodes[n].get('type') == 'thought']
        
        # Edge statistics by type
        edges_by_type = defaultdict(int)
        sim_weights = []
        
        for u, v, d in self.graph.edges(data=True):
            etype = d.get('edge_type', 'unknown')
            edges_by_type[etype] += 1
            if etype == 'similarity':
                sim_weights.append(d.get('weight', 0))
        
        # Thought type distribution
        thought_types = defaultdict(int)
        for node in thought_nodes:
            ttype = self.graph.nodes[node].get('thought_type', 'default')
            thought_types[ttype] += 1
        
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'num_solutions': len(self.solutions),
            'total_thoughts': len(self.thoughts),
            'num_communities': len(self.communities),
            'edges_by_type': dict(edges_by_type),
            'thought_type_distribution': dict(thought_types),
            'config': {
                'similarity_threshold': self.config.similarity_threshold,
                'top_k': self.config.top_k_neighbors
            }
        }
        
        if sim_weights:
            stats['similarity_stats'] = {
                'avg': float(np.mean(sim_weights)),
                'max': float(np.max(sim_weights)),
                'min': float(np.min(sim_weights))
            }
        
        # Per-solution stats
        stats['thoughts_per_solution'] = {
            key: len(sdata['thoughts'])
            for key, sdata in self.solutions.items()
        }
        
        # Community stats
        if self.communities:
            community_composition = {}
            for cid, nodes in self.communities.items():
                sol_dist = defaultdict(int)
                for node in nodes:
                    sol_key = self.thoughts[node]['solution_key']
                    sol_dist[sol_key] += 1
                community_composition[f'Community_{cid}'] = dict(sol_dist)
            stats['community_composition'] = community_composition
        
        return stats
    
    def export_data(
        self,
        graph_file: str = 'got_graph.gexf',
        json_file: str = 'got_data.json'
    ):
        """Export graph and metadata"""
        # Clean graph for export
        clean_graph = self._create_clean_graph()
        nx.write_gexf(clean_graph, graph_file)
        print(f"✓ Graph: {graph_file}")
        
        # JSON metadata
        stats = self.get_statistics()
        
        export_data = {
            'metadata': stats,
            'solutions': {
                key: {
                    'name': s['name'],
                    'model': s['model'],
                    'num_thoughts': len(s['thoughts'])
                }
                for key, s in self.solutions.items()
            },
            'thoughts': [
                {
                    'node_id': int(nid),
                    'thought_id': t['thought_id'],
                    'thought_type': t['thought_type'],
                    'solution_key': t['solution_key'],
                    'content': t['content'],
                    'confidence': t['confidence'],
                    'dependencies': t['dependencies'],
                    'community': self.graph.nodes[nid].get('community', -1)
                }
                for nid, t in self.thoughts.items()
            ]
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Metadata: {json_file}")
    
    def print_analysis(self):
        """Print detailed analysis"""
        print("\n" + "=" * 80)
        print("GRAPH OF THOUGHTS ANALYSIS")
        print("=" * 80)
        
        stats = self.get_statistics()
        
        print(f"\n📊 OVERVIEW:")
        print(f"  Total Nodes: {stats['total_nodes']}")
        print(f"  Total Edges: {stats['total_edges']}")
        print(f"  Solutions: {stats['num_solutions']}")
        print(f"  Thoughts: {stats['total_thoughts']}")
        print(f"  Communities: {stats['num_communities']}")
        
        print(f"\n🔗 EDGES BY TYPE:")
        for etype, count in stats['edges_by_type'].items():
            print(f"  {etype}: {count}")
        
        if 'similarity_stats' in stats:
            print(f"\n📈 SIMILARITY STATISTICS:")
            print(f"  Average: {stats['similarity_stats']['avg']:.3f}")
            print(f"  Max: {stats['similarity_stats']['max']:.3f}")
            print(f"  Min: {stats['similarity_stats']['min']:.3f}")
        
        print(f"\n🧠 THOUGHT TYPES:")
        for ttype, count in sorted(stats['thought_type_distribution'].items(), 
                                   key=lambda x: x[1], reverse=True):
            print(f"  {ttype}: {count}")
        
        print(f"\n📦 THOUGHTS PER SOLUTION:")
        for sol_key, count in stats['thoughts_per_solution'].items():
            print(f"  {sol_key}: {count}")
        
        if 'community_composition' in stats:
            print(f"\n🏘️ COMMUNITY COMPOSITION:")
            for comm_name, composition in stats['community_composition'].items():
                total = sum(composition.values())
                print(f"  {comm_name} ({total} thoughts):")
                for sol_key, count in sorted(composition.items(), 
                                            key=lambda x: x[1], reverse=True):
                    print(f"    - {sol_key}: {count}")
    
    # ==================== PRIVATE METHODS ====================
    
    def _next_id(self) -> int:
        nid = self.node_counter
        self.node_counter += 1
        return nid
    
    def _create_thought_node(
        self,
        thought: dict,
        parent: int,
        solution_key: str
    ) -> int:
        """Create thought node from chain_of_thought entry"""
        node_id = self._next_id()
        
        thought_id = thought['thought_id']
        thought_type = thought.get('thought_type', 'default')
        content = thought.get('content', '')
        confidence = thought.get('confidence', 0.5)
        
        self.graph.add_node(
            node_id,
            label=f"{thought_id}",
            text=content[:80],  # truncate for display
            type='thought',
            thought_type=thought_type,
            thought_id=thought_id,
            level=2,
            solution_key=solution_key,
            confidence=confidence
        )
        
        # Connect to solution root
        self.graph.add_edge(parent, node_id, edge_type='hierarchy')
        
        # Store full data
        self.thoughts[node_id] = {
            'thought_id': thought_id,
            'thought_type': thought_type,
            'content': content,
            'confidence': confidence,
            'dependencies': thought.get('dependencies', []),
            'reasoning_path': thought.get('reasoning_path', ''),
            'solution_key': solution_key,
            'node_id': node_id,
            'embedding': None  # will be computed later
        }
        
        return node_id
    
    def _hierarchical_layout(self) -> dict:
        """Create hierarchical layout (tree-like)"""
        pos = {}
        
        # Level-based positioning
        levels = defaultdict(list)
        for node in self.graph.nodes():
            level = self.graph.nodes[node].get('level', 0)
            levels[level].append(node)
        
        for level, nodes in levels.items():
            y = -level * 2  # vertical spacing
            x_spacing = 15 / max(len(nodes), 1)
            x_start = -7.5
            
            for i, node in enumerate(nodes):
                pos[node] = (x_start + i * x_spacing, y)
        
        return pos
    
    def _get_node_properties(self) -> Tuple[List, List]:
        """Get node colors and sizes"""
        colors = []
        sizes = []
        
        size_map = {
            'problem': 8000,
            'solution': 4000,
            'thought': 1200
        }
        
        for node in self.graph.nodes():
            ntype = self.graph.nodes[node].get('type', 'thought')
            
            # Color by thought type or community
            if ntype == 'thought':
                if 'community' in self.graph.nodes[node]:
                    # Color by community
                    comm = self.graph.nodes[node]['community']
                    palette = plt.cm.tab20(np.linspace(0, 1, 20))
                    colors.append(palette[comm % 20])
                else:
                    # Color by thought type
                    ttype = self.graph.nodes[node].get('thought_type', 'default')
                    colors.append(self.THOUGHT_TYPE_COLORS.get(ttype, '#95A5A6'))
            elif ntype == 'solution':
                colors.append('#2ECC71')
            else:  # problem
                colors.append('#E74C3C')
            
            sizes.append(size_map.get(ntype, 1200))
        
        return colors, sizes
    
    def _get_node_labels(self) -> dict:
        """Get node labels"""
        labels = {}
        for node in self.graph.nodes():
            ntype = self.graph.nodes[node].get('type')
            if ntype == 'thought':
                labels[node] = self.graph.nodes[node].get('thought_id', str(node))
            else:
                labels[node] = self.graph.nodes[node].get('label', str(node))
        return labels
    
    def _draw_edges_by_type(self, pos, ax):
        """Draw edges with different styles per type"""
        # Hierarchy edges (thin gray)
        hier_edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                     if d.get('edge_type') == 'hierarchy']
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=hier_edges,
            edge_color='#BDC3C7', alpha=0.3, width=0.5, 
            arrows=True, arrowsize=10, ax=ax
        )
        
        # Dependency edges (blue)
        dep_edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                    if d.get('edge_type') == 'dependency']
        nx.draw_networkx_edges(
            self.graph, pos, edgelist=dep_edges,
            edge_color='#3498DB', alpha=0.6, width=2,
            arrows=True, arrowsize=15, ax=ax
        )
        
        # Similarity edges (red, undirected)
        sim_edges = [(u, v, d) for u, v, d in self.graph.edges(data=True)
                    if d.get('edge_type') == 'similarity']
        
        if sim_edges:
            weights = [d['weight'] for _, _, d in sim_edges]
            edges_only = [(u, v) for u, v, _ in sim_edges]
            
            nx.draw_networkx_edges(
                self.graph, pos, edgelist=edges_only,
                edge_color=weights,
                edge_cmap=plt.cm.Reds,
                edge_vmin=0.6, edge_vmax=1.0,
                alpha=0.7, width=2.5, 
                arrows=False, ax=ax  # undirected
            )
    
    def _add_visualization_info(self, ax):
        """Add title and legend"""
        from matplotlib.patches import Patch
        
        stats = self.get_statistics()
        
        ax.set_title(
            f"Graph of Thoughts\n"
            f"Solutions: {stats['num_solutions']} | "
            f"Thoughts: {stats['total_thoughts']} | "
            f"Communities: {stats['num_communities']}",
            fontsize=20, fontweight='bold', pad=20
        )
        ax.axis('off')
        
        # Legend for thought types
        legend_items = [
            Patch(facecolor=color, label=ttype.replace('_', ' ').title())
            for ttype, color in self.THOUGHT_TYPE_COLORS.items()
            if ttype != 'default'
        ]
        
        ax.legend(handles=legend_items, loc='upper right', 
                 fontsize=9, title='Thought Types')
    
    def _create_clean_graph(self) -> nx.DiGraph:
        """Create serializable graph"""
        clean = nx.DiGraph()
        
        for node, data in self.graph.nodes(data=True):
            clean_data = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in data.items()
            }
            clean.add_node(node, **clean_data)
        
        for u, v, data in self.graph.edges(data=True):
            clean_data = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in data.items()
            }
            clean.add_edge(u, v, **clean_data)
        
        return clean


# ==================== DATA LOADING ====================

def load_multi_solution_data(data_dir: str = 'data') -> dict:
    """
    Load multiple solution files
    
    Expected file pattern: creasoning_*.json or solutions_*.json
    Each file should contain solution with chain_of_thought
    """
    data_path = Path(data_dir)
    
    problem_data = None
    all_solutions = []
    
    print("\n" + "=" * 80)
    print("LOADING MULTI-SOLUTION DATA")
    print("=" * 80)
    
    # Try different file patterns
    patterns = ['creasoning_*.json', 'solutions_*.json', 'cot_*.json']
    json_files = []
    
    for pattern in patterns:
        json_files.extend(data_path.glob(pattern))
    
    if not json_files:
        # Fallback: load all JSON files
        json_files = list(data_path.glob('*.json'))
    
    print(f"\n📁 Found {len(json_files)} JSON files")
    
    for file_path in sorted(json_files):
        print(f"\n  Loading: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        # Handle both list and single dict formats
        if isinstance(file_data, dict):
            file_data = [file_data]
        
        for solution in file_data:
            # Extract problem (use first occurrence)
            if problem_data is None:
                problem_data = {
                    'problem_restatement': solution.get('problem_restatement', 'Unknown Problem'),
                    'model': solution.get('model', 'unknown')
                }
            
            # Add solution
            if 'chain_of_thought' in solution:
                all_solutions.append(solution)
                print(f"    ✓ {solution.get('model', 'unknown')}: "
                      f"{len(solution.get('chain_of_thought', []))} thoughts")
    
    if not all_solutions:
        raise ValueError("No solutions with chain_of_thought found!")
    
    problem_data['solutions'] = all_solutions
    
    print(f"\n✓ Total solutions loaded: {len(all_solutions)}")
    print("=" * 80)
    
    return problem_data


# ==================== MAIN ====================

def graph_of_thoughts():
    """Main execution - Graph of Thoughts construction"""
    print("=" * 80)
    print("🌳 GRAPH OF THOUGHTS - MULTI-SOLUTION CoT MERGER 🌳")
    print("=" * 80)
    
    # Config
    config = GoTConfig(
        similarity_threshold=0.6,
        top_k_neighbors=3,
        community_resolution=1.2
    )
    
    # Load data
    problem_data = load_multi_solution_data('data')
    
    # Build Graph of Thoughts
    got = GraphOfThoughts(config)
    
    # Add problem root
    problem_node = got.add_problem_node(
        problem_data['problem_restatement']
    )
    print(f"\n✓ Problem node: {problem_node}")
    
    # Add all solutions with their CoT
    print("\n" + "-" * 80)
    print("ADDING SOLUTIONS WITH CHAIN OF THOUGHT")
    print("-" * 80)
    
    for idx, solution in enumerate(problem_data['solutions']):
        model_name = solution.get('model', f'model_{idx}')
        
        # Use approach_number if available, otherwise use index
        solution_id = solution.get('approach_number', 
                                   solution.get('solution_id', idx + 1))
        
        got.add_solution_with_cot(
            solution,
            model_name,
            solution_id,
            problem_node
        )
    
    # Compute embeddings
    print("\n" + "-" * 80)
    print("COMPUTING EMBEDDINGS")
    print("-" * 80)
    got.compute_embeddings()
    
    # Connect similar thoughts
    print("\n" + "-" * 80)
    print("CONNECTING SIMILAR THOUGHTS")
    print("-" * 80)
    got.connect_similar_thoughts()
    
    # Detect communities
    print("\n" + "-" * 80)
    print("DETECTING COMMUNITIES")
    print("-" * 80)
    got.detect_communities()
    
    # Analysis
    got.print_analysis()
    
    # Export
    print("\n" + "=" * 80)
    print("EXPORTING")
    print("=" * 80)
    got.export_data(
        'got_graph.gexf',
        'got_data.json'
    )
    
    # Visualize
    print("\n" + "=" * 80)
    print("VISUALIZING")
    print("=" * 80)
    
    # Create multiple visualizations
    layouts = ['hierarchical', 'spring']
    
    for layout in layouts:
        print(f"\n  Creating {layout} layout...")
        fig = got.visualize(layout=layout, figsize=(40, 30))
        filename = f'got_graph_{layout}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close()
    
    print("\n" + "=" * 80)
    print("✅ GRAPH OF THOUGHTS CONSTRUCTION COMPLETE!")
    print("=" * 80)
    print("\n📊 Output Files:")
    print("  - got_graph.gexf (import to Gephi/Cytoscape)")
    print("  - got_data.json (detailed metadata)")
    print("  - got_graph_hierarchical.png (tree view)")
    print("  - got_graph_spring.png (similarity view)")
    print("\n💡 Key Insights:")
    stats = got.get_statistics()
    print(f"  - Merged {stats['num_solutions']} solution chains")
    print(f"  - Total {stats['total_thoughts']} unique thoughts")
    print(f"  - Found {stats['edges_by_type'].get('similarity', 0)} cross-solution connections")
    print(f"  - Identified {stats['num_communities']} thought communities")
    print("=" * 80)


