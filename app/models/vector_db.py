from typing import List
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class VectorDB:
    # File path for saving embeddings
    EMBEDDINGS_FILE = "data/embeddings.pkl"
    
    def __init__(self):
        # Initialize the sentence transformer model for embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension of the embeddings from the model
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(self.EMBEDDINGS_FILE), exist_ok=True)
        
        # Try to load existing index and documents
        if os.path.exists(self.EMBEDDINGS_FILE):
            self.load_embeddings()
        else:
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
        
        # Save embeddings to disk
        self.save_embeddings()
    
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
    
    def save_embeddings(self) -> None:
        """
        Save the index and documents to disk
        """
        # Convert index to bytes for serialization
        index_bytes = faiss.serialize_index(self.index)
        
        # Save both index bytes and documents
        with open(self.EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump({
                'index_bytes': index_bytes,
                'documents': self.documents
            }, f)
    
    def load_embeddings(self) -> None:
        """
        Load the index and documents from disk
        """
        try:
            with open(self.EMBEDDINGS_FILE, 'rb') as f:
                data = pickle.load(f)
                
            # Deserialize the index
            self.index = faiss.deserialize_index(data['index_bytes'])
            self.documents = data['documents']
            print(f"Loaded {len(self.documents)} documents from disk")
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            # Initialize empty if loading fails
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = []