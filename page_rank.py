"""
Graph-based Answer Ranker using HuggingFace embeddings (sentence-transformers).

Features:
- Uses `mixedbread-ai/mxbai-embed-large-v1` by default (change EMBED_MODEL if you prefer).
- Automatic device detection (GPU/CPU).
- Batch embedding and chunking for long texts.
- L2 normalization of embeddings.
- In-memory caching of embeddings.
- PageRank / Eigenvector centrality ranking over a similarity graph.

Requirements:
pip install -U sentence-transformers torch numpy requests scipy
"""

import os
import sys
import json
import time
from glob import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
from scipy.sparse.linalg import eigs

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
except Exception as e:
    raise ImportError(
        "Missing dependency: sentence-transformers. Install with:\n"
        "  pip install sentence-transformers torch\n\n"
        f"Underlying error: {e}"
    )

# -------------------------
# Configuration (change here if desired)
# -------------------------
EMBED_MODEL = "mixedbread-ai/mxbai-embed-large-v1"  # recommended for scientific/reasoning text
MAX_CHARS = 32000         # maximum chars to accept per piece before truncation
CHUNK_SIZE = 3000         # chunk size in chars for long texts (simple chunking)
BATCH_SIZE = 16           # batch size for model.encode
EMBED_DTYPE = np.float32  # dtype for embeddings


# -------------------------
# Embedding provider
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
        # prefer GPU if available
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # MPS (Apple silicon) support
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
            # warm-up a tiny encoding to ensure model is ready
            _ = self._model.encode("warmup", show_progress_bar=False)
            print(f"[INFO] Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}'.\n"
                f"Install dependencies and ensure model name is correct.\nUnderlying error: {e}"
            )

    def _chunk_text(self, text: str) -> List[str]:
        """Simple character-based chunking preserving order."""
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
        """L2 normalize rows"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(EMBED_DTYPE)

    def embed(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Return a single embedding vector for the provided text.
        If text is longer than MAX_CHARS, it will be truncated (configurable).
        For very long inputs, text is chunked, embedded per-chunk, then averaged and normalized.
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        # Truncate extremely long texts
        if len(text) > MAX_CHARS:
            print(f"[WARN] Input longer than {MAX_CHARS} chars — truncating.")
            text = text[:MAX_CHARS]

        if use_cache and text in self._cache:
            return self._cache[text]

        # Chunk if necessary
        chunks = self._chunk_text(text)

        # Batch encode chunks
        embeddings_list = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            embs = self._model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False)
            # ensure dtype
            embs = embs.astype(EMBED_DTYPE)
            embeddings_list.append(embs)

        if len(embeddings_list) == 0:
            raise RuntimeError("No embeddings produced for input.")

        embeddings = np.vstack(embeddings_list)
        # Aggregate chunk embeddings (mean) then normalize
        agg = embeddings.mean(axis=0, keepdims=False)
        agg = agg.astype(EMBED_DTYPE)
        agg = self._normalize(agg.reshape(1, -1))[0]

        if use_cache:
            self._cache[text] = agg

        return agg

    def embed_batch(self, texts: List[str], use_cache: bool = True) -> List[np.ndarray]:
        """Embed a list of texts; uses caching and batching inside embed()."""
        results = []
        for t in texts:
            results.append(self.embed(t, use_cache=use_cache))
        return results

    def clear_cache(self):
        self._cache.clear()


# -------------------------
# GraphBasedRanker using EmbeddingProvider
# -------------------------
class GraphBasedRanker:
    """
    Graph-based ranking using PageRank / Eigenvector Centrality.
    Uses EmbeddingProvider (sentence-transformers) instead of Ollama.
    """

    def __init__(self, embed_provider: Optional[EmbeddingProvider] = None, model_name: str = EMBED_MODEL):
        self.model_name = model_name
        self.embed_provider = embed_provider or EmbeddingProvider(model_name)
        # no Ollama usage

    def get_embedding(self, text: str) -> np.ndarray:
        """Wrapper to fetch normalized embedding from provider."""
        return self.embed_provider.embed(text)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        if vec1 is None or vec2 is None:
            return 0.0
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        if vec1.shape != vec2.shape:
            raise ValueError(f"Vector shapes do not match: {vec1.shape} vs {vec2.shape}")
        dot = float(np.dot(vec1, vec2))
        n1 = float(np.linalg.norm(vec1))
        n2 = float(np.linalg.norm(vec2))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        sim = dot / (n1 * n2)
        return float(np.clip(sim, -1.0, 1.0))

    def build_similarity_graph(self, embeddings: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
        n = len(embeddings)
        if n == 0:
            raise ValueError("Embeddings list is empty.")
        mat = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    sim = self.cosine_similarity(embeddings[i], embeddings[j])
                    if sim > threshold:
                        mat[i, j] = sim
                        mat[j, i] = sim
                except Exception as e:
                    print(f"[WARN] Similarity calc failed for ({i},{j}): {e}")
        return mat

    def pagerank(self, adjacency: np.ndarray, damping: float = 0.85,
                 max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        n = adjacency.shape[0]
        if n == 0:
            return np.array([])
        row_sum = adjacency.sum(axis=1)
        if np.all(row_sum == 0):
            print("[WARN] Graph is fully disconnected. Returning uniform scores.")
            return np.ones(n, dtype=np.float32) / n
        row_sum[row_sum == 0] = 1.0
        trans = adjacency / row_sum[:, None]
        scores = np.ones(n, dtype=np.float32) / n
        for it in range(max_iter):
            prev = scores.copy()
            scores = (1 - damping) / n + damping * (trans.T @ scores)
            # normalize to sum=1
            s = scores.sum()
            if s != 0:
                scores = scores / s
            diff = np.linalg.norm(scores - prev, 1)
            if diff < tol:
                print(f"[INFO] PageRank converged in {it+1} iterations (diff={diff:.2e}).")
                break
        else:
            print(f"[WARN] PageRank did not converge in {max_iter} iterations.")
        return scores

    def eigenvector_centrality(self, adjacency: np.ndarray, max_iter: int = 100) -> np.ndarray:
        n = adjacency.shape[0]
        if n == 0:
            return np.array([])
        if np.all(adjacency == 0):
            print("[WARN] Adjacency matrix is zero. Returning uniform centrality.")
            return np.ones(n, dtype=np.float32) / n
        try:
            vals, vecs = eigs(adjacency.astype(np.float64), k=1, which='LM', maxiter=max_iter)
            principal = np.abs(vecs[:, 0].real)
            s = principal.sum()
            if s > 0:
                return (principal / s).astype(np.float32)
            else:
                return np.ones(n, dtype=np.float32) / n
        except Exception as e:
            print(f"[WARN] Eigenvector centrality failed: {e}. Returning uniform scores.")
            return np.ones(n, dtype=np.float32) / n

    def rank_answers(self, answers: List[str], method: str = "pagerank",
                     threshold: float = 0.5, damping: float = 0.85) -> List[Tuple[int, float, str]]:
        if not answers:
            raise ValueError("Answers list is empty.")
        print(f"[INFO] Fetching embeddings for {len(answers)} answers...")
        embeddings: List[np.ndarray] = []
        valid_indices: List[int] = []
        for i, ans in enumerate(answers):
            print(f"[INFO] Processing answer {i+1}/{len(answers)}...", end=" ")
            try:
                emb = self.get_embedding(ans)
                embeddings.append(emb)
                valid_indices.append(i)
                print("OK")
                time.sleep(0.01)
            except Exception as e:
                print(f"FAILED: {e}")
        if len(embeddings) == 0:
            raise ValueError("No embeddings were retrieved. Please check the model and internet connection.")
        if len(embeddings) < len(answers):
            print(f"[WARN] {len(answers) - len(embeddings)} answers were skipped due to errors.")
        print("[INFO] Building similarity graph...")
        sim_graph = self.build_similarity_graph(embeddings, threshold)
        print(f"[INFO] Number of edges: {int(np.count_nonzero(sim_graph) // 2)}")
        print(f"[INFO] Ranking using method: {method}")
        if method.lower() == "pagerank":
            scores = self.pagerank(sim_graph, damping)
        elif method.lower() == "eigenvector":
            scores = self.eigenvector_centrality(sim_graph)
        else:
            raise ValueError("Invalid method. Use 'pagerank' or 'eigenvector'.")
        results = [(valid_indices[i], float(scores[i]), answers[valid_indices[i]]) for i in range(len(embeddings))]
        results.sort(key=lambda x: x[1], reverse=True)
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
def page_rank():
    print("=" * 60)
    print("🚀 Graph-based Answer Ranker (HF embeddings)")
    print("=" * 60)

    test_answers: List[str] = []
    files = glob("data_1/creasoning_*.json")
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
        ranker = GraphBasedRanker(embed_provider=provider)
    except Exception as e:
        print(f"[ERROR] Failed to initialize embedding provider or ranker: {e}")
        return

    try:
        results = ranker.rank_answers(
            test_answers,
            method="pagerank",
            threshold=0.3,
            damping=0.85
        )
        print("\n" + "=" * 60)
        print("🏆 Results (Top 5)")
        print("=" * 60)
        for rank, (idx, score, answer) in enumerate(results[:5], 1):
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"\n#{rank} - score: {score:.4f} (index: {idx})")
            print(f"   {preview}")
        print("\n" + "=" * 60)
        print(f"[INFO] Ranking completed for {len(results)} answers.")
    except Exception as e:
        print(f"[ERROR] Ranking failed: {e}")
        import traceback
        traceback.print_exc()



