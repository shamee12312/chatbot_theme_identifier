import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import chromadb
from chromadb.config import Settings
import uuid
from sentence_transformers import SentenceTransformer
import pickle
import os
import json

class VectorStore:
    """
    Handles vector storage and similarity search using ChromaDB
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.model_name = "all-MiniLM-L6-v2"  # Lightweight but effective model
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection_name = "document_embeddings"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            print("Using fallback embedding method")
            self.embedding_model = None
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        # Document storage
        self.documents = {}
    
    def add_document(self, doc_data: Dict[str, Any]) -> bool:
        """
        Add a document to the vector store
        
        Args:
            doc_data: Processed document data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            doc_id = doc_data['doc_id']
            
            # Store document data
            self.documents[doc_id] = doc_data
            
            # Prepare text chunks for embedding
            chunks = self._prepare_text_chunks(doc_data)
            
            # Generate embeddings
            embeddings = self._generate_embeddings(chunks)
            
            # Prepare metadata and IDs
            chunk_ids = []
            metadatas = []
            texts = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}_{i}"
                chunk_ids.append(chunk_id)
                texts.append(chunk['text'])
                
                metadata = {
                    'doc_id': doc_id,
                    'document_name': doc_data['original_name'],
                    'chunk_index': i,
                    'page_number': chunk['page_number'],
                    'paragraph_number': chunk.get('paragraph_number', 0),
                    'sentence_number': chunk.get('sentence_number', 0),
                    'word_count': chunk['word_count'],
                    'document_type': doc_data['document_type']
                }
                metadatas.append(metadata)
            
            # Add to ChromaDB
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            print(f"Successfully added document {doc_data['original_name']} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error adding document to vector store: {str(e)}")
            return False
    
    def search(self, query: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar documents/chunks
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embeddings([{'text': query}])[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Process results
            search_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'document_data': self.documents.get(results['metadatas'][0][i]['doc_id'])
                }
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            print(f"Error searching vector store: {str(e)}")
            return []
    
    def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of document chunks
        """
        try:
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=['metadatas', 'documents']
            )
            
            chunks = []
            for i in range(len(results['ids'])):
                chunk = {
                    'id': results['ids'][i],
                    'text': results['documents'][i],
                    'metadata': results['metadatas'][i]
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"Error getting document chunks: {str(e)}")
            return []
    
    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document from the vector store
        
        Args:
            doc_id: Document ID to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"doc_id": doc_id},
                include=['metadatas']
            )
            
            chunk_ids = results['ids']
            
            if chunk_ids:
                # Delete from ChromaDB
                self.collection.delete(ids=chunk_ids)
                
                # Remove from documents cache
                if doc_id in self.documents:
                    del self.documents[doc_id]
                
                print(f"Successfully removed document {doc_id}")
                return True
            else:
                print(f"Document {doc_id} not found")
                return False
                
        except Exception as e:
            print(f"Error removing document: {str(e)}")
            return False
    
    def _prepare_text_chunks(self, doc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare text chunks for embedding
        
        Args:
            doc_data: Document data
            
        Returns:
            List of text chunks with metadata
        """
        chunks = []
        
        for page in doc_data['pages']:
            page_num = page['page_number']
            
            # Create chunks at paragraph level
            for paragraph in page['paragraphs']:
                # Skip very short paragraphs
                if paragraph['word_count'] < 10:
                    continue
                
                chunk = {
                    'text': paragraph['text'],
                    'page_number': page_num,
                    'paragraph_number': paragraph['paragraph_number'],
                    'word_count': paragraph['word_count']
                }
                chunks.append(chunk)
                
                # For longer paragraphs, also create sentence-level chunks
                if paragraph['word_count'] > 100:
                    for sentence in paragraph['sentences']:
                        if sentence['word_count'] > 5:
                            sentence_chunk = {
                                'text': sentence['text'],
                                'page_number': page_num,
                                'paragraph_number': paragraph['paragraph_number'],
                                'sentence_number': sentence['sentence_number'],
                                'word_count': sentence['word_count']
                            }
                            chunks.append(sentence_chunk)
        
        return chunks
    
    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Generate embeddings for text chunks
        
        Args:
            chunks: