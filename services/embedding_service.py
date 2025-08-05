from typing import List, Dict, Any
import gc

class EmbeddingService:
    def __init__(self):
        """Initialize with lazy loading to save memory"""
        self.model = None
        self.dimension = 384
    
    def _load_model(self):
        """Lazy load model only when needed"""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            # Force garbage collection
            gc.collect()
    
    def encode_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> List[List[float]]:
        """Generate embeddings with memory optimization"""
        try:
            self._load_model()
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings = self.model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
                all_embeddings.extend(embeddings.tolist())
                gc.collect()
            
            return all_embeddings
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def encode_single_text(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        try:
            self._load_model()
            embedding = self.model.encode([text], convert_to_tensor=False, show_progress_bar=False)[0]
            gc.collect()
            return embedding.tolist()
        except Exception as e:
            raise Exception(f"Failed to generate embedding: {str(e)}")