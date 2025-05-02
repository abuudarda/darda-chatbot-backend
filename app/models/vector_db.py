from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self):
        # Initialize the sentence transformer model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension of the embeddings from the model
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
        
    def add_texts(self, texts: List[str]) -> None:
        """
        Add text chunks to the vector database
        
        Args:
            texts (List[str]): List of text chunks to add
        """
        if not texts:
            return
        
        # Generate embeddings
        embeddings = self.model.encode(texts)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store documents
        self.documents.extend(texts)
    
    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        """
        Search for the most similar documents to the query
        
        Args:
            query (str): The search query
            k (int): Number of results to return
            
        Returns:
            List[str]: List of the most similar documents
        """
        if len(self.documents) == 0:
            return []
            
        # Generate query embedding
        query_embedding = self.model.encode([query])
        
        # Search in the index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            k=min(k, len(self.documents))
        )
        
        # Return the most similar documents
        results = [self.documents[idx] for idx in indices[0]]
        return results