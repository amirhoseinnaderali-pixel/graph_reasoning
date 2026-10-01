from typing import Sequence
import numpy as np

class EmbeddingProvider:
    """Configurable embeddings; TF-IDF keeps the core usable without model downloads."""
    def __init__(self, backend="tfidf", model_name="mixedbread-ai/mxbai-embed-large-v1", max_features=2048):
        self.backend,self.model_name,self.max_features=backend,model_name,max_features
        self.model=None
        if backend=="sentence_transformers":
            try: from sentence_transformers import SentenceTransformer
            except ImportError as e: raise ImportError("Install gcr-llm[embeddings] to use sentence-transformers.") from e
            self.model=SentenceTransformer(model_name)
        elif backend!="tfidf": raise ValueError(f"Unknown embedding backend: {backend}")
    def encode(self, texts: Sequence[str]):
        if not texts: return np.empty((0,0),dtype=np.float32)
        if self.backend=="sentence_transformers":
            return np.asarray(self.model.encode(list(texts),convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False),dtype=np.float32)
        from sklearn.feature_extraction.text import TfidfVectorizer
        return TfidfVectorizer(max_features=self.max_features,norm="l2").fit_transform(list(texts)).toarray().astype(np.float32)
