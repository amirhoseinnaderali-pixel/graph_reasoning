"""
Clustering-based Answer Ranker using HuggingFace embeddings (sentence-transformers).

Strategy: Embedding + Clustering + Centroid Selection
- Embeds all answers
- Clusters them using K-Means or HDBSCAN
- Finds the answer closest to each cluster centroid
- Ranks clusters by size or internal coherence

Features:
- Uses `mixedbread-ai/mxbai-embed-large-v1` by default
- Automatic device detection (GPU/CPU)
- Batch embedding and chunking for long texts
- L2 normalization of embeddings
- In-memory caching of embeddings
- Multiple clustering algorithms (K-Means, HDBSCAN)
- Visualization of clusters with matplotlib

Requirements:
pip install -U sentence-transformers torch numpy scikit-learn hdbscan matplotlib
"""

import os
import sys
import json
import time
from glob import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError(
        "Missing dependency: sentence-transformers. Install with:\n"
        "  pip install sentence-transformers torch\n\n"
        f"Underlying error: {e}"
    )

# Try to import HDBSCAN (optional)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("[WARN] HDBSCAN not available. Install with: pip install hdbscan")

# -------------------------
# Configuration
# -------------------------
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
MAX_CHARS = 32000
CHUNK_SIZE = 3000
BATCH_SIZE = 16
EMBED_DTYPE = np.float32


# -------------------------
# Embedding provider (from original code)
# -------------------------
class EmbeddingProvider:
    """
    Wrapper for sentence-transformers model providing:
      - device auto-detection
      - chunking for long texts
      - batch embedding
      - L2 normalization
      - simple in-memory caching
    """
    def __init__(self, model_name: str = EMBED_MODEL, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or self._detect_device()
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}
        self._load_model()

    def _detect_device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        try:
            if getattr(torch, "has_mps", False) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def _load_model(self):
        print(f"[INFO] Loading embedding model '{self.model_name}' on device '{self.device}'...")
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
            _ = self._model.encode("warmup", show_progress_bar=False)
            print(f"[INFO] Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}'.\n"
                f"Install dependencies and ensure model name is correct.\nUnderlying error: {e}"
            )

    def _chunk_text(self, text: str) -> List[str]:
        if len(text) <= CHUNK_SIZE:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            chunks.append(text[start:end])
            start = end
        return chunks

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(EMBED_DTYPE)

    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        if len(text) > MAX_CHARS:
            print(f"[WARN] Input longer than {MAX_CHARS} chars — truncating.")
            text = text[:MAX_CHARS]

        if use_cache and text in self._cache:
            return self._cache[text]

        chunks = self._chunk_text(text)
        embeddings_list = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            embs = self._model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
            embs = embs.astype(EMBED_DTYPE)
            embeddings_list.append(embs)

        if len(embeddings_list) == 0:
            raise RuntimeError("No embeddings produced for input.")

        embeddings = np.vstack(embeddings_list)
        agg = embeddings.mean(axis=0, keepdims=False)
        agg = agg.astype(EMBED_DTYPE)
        agg = self._normalize(agg.reshape(1, -1))[0]

        if use_cache:
            self._cache[text] = agg

        return agg

    def embed_batch(self, texts: List[str], use_cache: bool = True) -> List[np.ndarray]:
        results = []
        for t in texts:
            results.append(self.embed(t, use_cache=use_cache))
        return results

    def clear_cache(self):
        self._cache.clear()


# -------------------------
# ClusteringBasedRanker
# -------------------------
class ClusteringBasedRanker:
    """
    Clustering-based ranking: Embed → Cluster → Find Centroid Representatives
    
    Methods:
    - kmeans: Traditional K-Means clustering
    - hdbscan: Density-based clustering (automatically finds number of clusters)
    """

    def __init__(self, embed_provider: Optional[EmbeddingProvider] = None, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self.embed_provider = embed_provider or EmbeddingProvider(model_name)

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for all texts and stack them into a matrix."""
        print(f"[INFO] Fetching embeddings for {len(texts)} answers...")
        embeddings = []
        for i, text in enumerate(texts):
            print(f"[INFO] Processing answer {i+1}/{len(texts)}...", end=" ")
            try:
                emb = self.embed_provider.embed(text)
                embeddings.append(emb)
                print("OK")
            except Exception as e:
                print(f"FAILED: {e}")
                # Use zero vector as fallback
                if embeddings:
                    embeddings.append(np.zeros_like(embeddings[0]))
                else:
                    # If first embedding fails, we can't determine dimension
                    raise ValueError(f"First embedding failed: {e}")
        
        return np.vstack(embeddings)

    def find_optimal_k(self, embeddings: np.ndarray, min_k: int = 2, max_k: int = 10) -> int:
        """Find optimal number of clusters using silhouette score."""
        if len(embeddings) < min_k:
            return max(2, len(embeddings) // 2)
        
        max_k = min(max_k, len(embeddings) - 1)
        best_k = min_k
        best_score = -1
        
        print(f"[INFO] Finding optimal K (testing {min_k} to {max_k})...")
        for k in range(min_k, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                # Need at least 2 clusters for silhouette score
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(embeddings, labels)
                    print(f"[INFO] K={k}: silhouette={score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_k = k
            except Exception as e:
                print(f"[WARN] K={k} failed: {e}")
        
        print(f"[INFO] Optimal K: {best_k} (score: {best_score:.4f})")
        return best_k

    def kmeans_clustering(self, embeddings: np.ndarray, n_clusters: Optional[int] = None,
                          auto_k: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform K-Means clustering.
        
        Returns:
            labels: cluster assignment for each answer
            centroids: cluster centroids
        """
        if auto_k and n_clusters is None:
            n_clusters = self.find_optimal_k(embeddings)
        elif n_clusters is None:
            n_clusters = min(5, max(2, len(embeddings) // 3))
        
        n_clusters = min(n_clusters, len(embeddings))
        
        print(f"[INFO] Running K-Means with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
        
        return labels, centroids

    def hdbscan_clustering(self, embeddings: np.ndarray, min_cluster_size: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform HDBSCAN clustering (density-based).
        
        Returns:
            labels: cluster assignment for each answer (-1 for noise)
            centroids: computed centroids for each cluster
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        print(f"[INFO] Running HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        labels = clusterer.fit_predict(embeddings)
        
        # Compute centroids manually
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise
        
        centroids = []
        for label in unique_labels:
            cluster_points = embeddings[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
        
        if len(centroids) == 0:
            print("[WARN] No clusters found, using single cluster")
            return np.zeros(len(embeddings), dtype=int), embeddings.mean(axis=0, keepdims=True)
        
        centroids = np.vstack(centroids)
        print(f"[INFO] Found {len(centroids)} clusters ({np.sum(labels == -1)} noise points)")
        
        return labels, centroids

    def find_centroid_representatives(self, embeddings: np.ndarray, labels: np.ndarray,
                                     centroids: np.ndarray) -> List[Tuple[int, int, float]]:
        """
        Find the answer closest to each cluster centroid.
        
        Returns:
            List of (cluster_id, answer_index, distance_to_centroid)
        """
        representatives = []
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise in HDBSCAN
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_embeddings = embeddings[cluster_indices]
            centroid = centroids[cluster_id]
            
            # Find closest point to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            closest_idx_in_cluster = np.argmin(distances)
            closest_idx_global = cluster_indices[closest_idx_in_cluster]
            closest_distance = distances[closest_idx_in_cluster]
            
            representatives.append((int(cluster_id), int(closest_idx_global), float(closest_distance)))
        
        return representatives

    def visualize_clusters(self, embeddings: np.ndarray, labels: np.ndarray,
                          centroids: np.ndarray, representatives: List[Tuple[int, int, float]],
                          answers: List[str], output_path: str = "clusters_visualization.png"):
        """
        Visualize clusters in 2D using PCA.
        
        Args:
            embeddings: All embeddings (N x D)
            labels: Cluster labels for each point
            centroids: Cluster centroids
            representatives: List of (cluster_id, answer_index, distance)
            answers: Original answer texts (for labeling)
            output_path: Where to save the visualization
        """
        print(f"[INFO] Creating visualization...")
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        centroids_2d = pca.transform(centroids)
        
        # Get representative indices
        rep_indices = {rep[1]: rep[0] for rep in representatives}  # answer_idx -> cluster_id
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Get unique labels (excluding noise -1 if present)
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        # Color palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for idx, cluster_id in enumerate(unique_labels):
            cluster_mask = labels == cluster_id
            cluster_points = embeddings_2d[cluster_mask]
            
            # Plot cluster points
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      c=[colors[idx]], alpha=0.6, s=100,
                      label=f'Cluster {cluster_id} ({np.sum(cluster_mask)} items)',
                      edgecolors='black', linewidth=0.5)
            
            # Plot centroid
            ax.scatter(centroids_2d[cluster_id, 0], centroids_2d[cluster_id, 1],
                      c=[colors[idx]], marker='*', s=500,
                      edgecolors='black', linewidth=2, zorder=10)
        
        # Plot noise points if any (HDBSCAN)
        if -1 in labels:
            noise_mask = labels == -1
            noise_points = embeddings_2d[noise_mask]
            ax.scatter(noise_points[:, 0], noise_points[:, 1],
                      c='gray', alpha=0.3, s=50, marker='x',
                      label=f'Noise ({np.sum(noise_mask)} items)')
        
        # Highlight representatives with annotations
        for answer_idx, cluster_id in rep_indices.items():
            x, y = embeddings_2d[answer_idx]
            
            # Draw a ring around representative
            circle = plt.Circle((x, y), radius=0.15, color='red',
                              fill=False, linewidth=3, zorder=5)
            ax.add_patch(circle)
            
            # Add text label with preview
            preview = answers[answer_idx][:40] + "..." if len(answers[answer_idx]) > 40 else answers[answer_idx]
            ax.annotate(f'Rep C{cluster_id}',
                       xy=(x, y), xytext=(10, 10),
                       textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red'),
                       fontsize=8, zorder=15)
        
        # Formatting
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        ax.set_title('Clustering Visualization: Embeddings in 2D Space\n' +
                    '★ = Centroid | ⭕ = Representative Answer',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Visualization saved to: {output_path}")
        plt.close()

    def rank_answers(self, answers: List[str], method: str = "kmeans",
                     n_clusters: Optional[int] = None, auto_k: bool = True,
                     rank_by: str = "size", visualize: bool = True,
                     viz_path: str = "clusters_visualization.png") -> List[Tuple[int, float, str, int]]:
        """
        Rank answers using clustering approach.
        
        Args:
            answers: List of answer strings
            method: 'kmeans' or 'hdbscan'
            n_clusters: Number of clusters (for kmeans, ignored if auto_k=True)
            auto_k: Automatically find optimal K (for kmeans)
            rank_by: 'size' (cluster size) or 'coherence' (inverse avg distance)
            visualize: Whether to create a visualization
            viz_path: Path to save visualization
        
        Returns:
            List of (answer_index, score, answer_text, cluster_id)
        """
        if not answers:
            raise ValueError("Answers list is empty.")
        
        # Get embeddings
        embeddings = self.get_embeddings(answers)
        
        # Cluster
        if method.lower() == "kmeans":
            labels, centroids = self.kmeans_clustering(embeddings, n_clusters, auto_k)
        elif method.lower() == "hdbscan":
            labels, centroids = self.hdbscan_clustering(embeddings)
        else:
            raise ValueError("Invalid method. Use 'kmeans' or 'hdbscan'.")
        
        # Find representatives
        print("[INFO] Finding centroid representatives...")
        representatives = self.find_centroid_representatives(embeddings, labels, centroids)
        
        # Compute cluster scores
        print(f"[INFO] Ranking clusters by: {rank_by}")
        cluster_scores = {}
        
        for cluster_id, rep_idx, dist in representatives:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            if rank_by == "size":
                # Larger clusters = higher score
                score = float(cluster_size)
            elif rank_by == "coherence":
                # Lower average distance to centroid = higher score
                cluster_embeddings = embeddings[cluster_indices]
                centroid = centroids[cluster_id]
                avg_dist = np.mean(np.linalg.norm(cluster_embeddings - centroid, axis=1))
                score = 1.0 / (avg_dist + 1e-6)  # Inverse distance
            else:
                score = float(cluster_size)  # Default to size
            
            cluster_scores[cluster_id] = score
        
        # Build results
        results = []
        for cluster_id, rep_idx, dist in representatives:
            score = cluster_scores[cluster_id]
            results.append((rep_idx, score, answers[rep_idx], cluster_id))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Create visualization if requested
        if visualize:
            try:
                self.visualize_clusters(embeddings, labels, centroids, representatives, answers, viz_path)
            except Exception as e:
                print(f"[WARN] Visualization failed: {e}")
        
        print("[INFO] Ranking completed.")
        return results


# -------------------------
# Utility functions
# -------------------------
def load_creasoning_file(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON decode error ({path}): {e}")
        return []
    answers: List[str] = []
    if isinstance(data, list):
        for item in data:
            answers.append(json.dumps(item, ensure_ascii=False))
    else:
        answers.append(json.dumps(data, ensure_ascii=False))
    return answers


# -------------------------
# CLI main
# -------------------------
def k_means():
    print("=" * 60)
    print("🎯 Clustering-based Answer Ranker (HF embeddings)")
    print("=" * 60)

    test_answers: List[str] = []
    files = glob("data/creasoning_*.json")
    if not files:
        print("[ERROR] No files found matching 'data/creasoning_*.json'")
        return
    for fp in files:
        print(f"[INFO] Loading: {fp}")
        answers = load_creasoning_file(fp)
        test_answers.extend(answers)
        print(f"[INFO] Loaded {len(answers)} answers from {fp}")
    if not test_answers:
        print("[ERROR] No answers were loaded. Exiting.")
        return
    print(f"[INFO] Total answers: {len(test_answers)}")

    try:
        provider = EmbeddingProvider(model_name=EMBED_MODEL)
        ranker = ClusteringBasedRanker(embed_provider=provider)
    except Exception as e:
        print(f"[ERROR] Failed to initialize embedding provider or ranker: {e}")
        return

    try:
        # Try K-Means with automatic K selection
        results = ranker.rank_answers(
            test_answers,
            method="kmeans",
            auto_k=True,
            rank_by="size",  # or "coherence"
            visualize=True,
            viz_path="clusters_visualization.png"
        )
        
        print("\n" + "=" * 60)
        print("🏆 Results (Top 5 Cluster Representatives)")
        print("=" * 60)
        for rank, (idx, score, answer, cluster_id) in enumerate(results[:5], 1):
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"\n#{rank} - Cluster {cluster_id} | Score: {score:.2f} | Index: {idx}")
            print(f"   {preview}")
        
        print("\n" + "=" * 60)
        print(f"[INFO] Ranking completed. Found {len(results)} cluster representatives.")
        
    except Exception as e:
        print(f"[ERROR] Ranking failed: {e}")
        import traceback
        traceback.print_exc()


