import openai
import os
from typing import List, Dict, Any, Optional
import json
from collections import defaultdict
import re

class QueryEngine:
    """
    Handles query processing, document search, and answer generation
    """
    
    def __init__(self, vector_store, theme_analyzer):
        self.vector_store = vector_store
        self.theme_analyzer = theme_analyzer
        
        # Initialize OpenAI client
        self.openai_client = None
        self._initialize_openai()
        
        # Query processing settings
        self.max_chunks_per_doc = 5
        self.min_relevance_score = 0.3
        self.max_documents = 20
    
    def _initialize_openai(self):
        """Initialize OpenAI client if API key is available"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                openai.api_key = api_key
                self.openai_client = openai
                print("Query Engine: OpenAI client initialized")
            else:
                print("Query Engine: OpenAI API key not found. Using fallback methods.")
        except Exception as e:
            print(f"Query Engine: Error initializing OpenAI: {str(e)}")
            self.openai_client = None
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return comprehensive results
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing document responses and themes
        """
        try:
            # Step 1: Search for relevant document chunks
            search_results = self.vector_store.search(query, n_results=50)
            
            if not search_results:
                return {
                    'query': query,
                    'document_responses': [],
                    'themes': [],
                    'message': 'No relevant documents found for your query.'
                }
            
            # Step 2: Group results by document
            doc_groups = self._group_by_document(search_results)
            
            # Step 3: Generate answers for each document
            document_responses = []
            for doc_id, chunks in doc_groups.items():
                doc_response = self._generate_document_response(query, chunks)
                if doc_response:
                    document_responses.append(doc_response)
            
            # Step 4: Sort by relevance and limit results
            document_responses.sort(key=lambda x: x['relevance_score'], reverse=True)
            document_responses = document_responses[:self.max_documents]
            
            # Step 5: Identify themes across all document responses
            themes = self.theme_analyzer.analyze_themes(query, document_responses)
            
            return {
                'query': query,
                'document_responses': document_responses,
                'themes': themes,
                'total_documents_searched': len(doc_groups),
                'total_chunks_analyzed': len(search_results)
            }
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return {
                'query': query,
                'document_responses': [],
                'themes': [],
                'error': str(e)
            }
    
    def _group_by_document(self, search_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group search results by document ID
        
        Args:
            search_results: List of search results
            
        Returns:
            Dictionary mapping doc_id to list of chunks
        """
        doc_groups = defaultdict(list)
        
        for result in search_results:
            if result['similarity_score'] >= self.min_relevance_score:
                doc_id = result['metadata']['doc_id']
                doc_groups[doc_id].append(result)
        
        # Limit chunks per document and sort by relevance
        for doc_id in doc_groups:
            doc_groups[doc_id].sort(key=lambda x: x['similarity_score'], reverse=True)
            doc_groups[doc_id] = doc_groups[doc_id][:self.max_chunks_per_doc]
        
        return dict(doc_groups)
    
    def _generate_document_response(self, query: str, chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Generate a response for a specific document based on relevant chunks
        
        Args:
            query: User query
            chunks: Relevant chunks from the document
            
        Returns:
            Document response dictionary
        """
        try:
            if not chunks:
                return None
            
            # Get document metadata
            doc_metadata = chunks[0]['metadata']
            doc_data = chunks[0]['document_data']
            
            # Combine relevant text chunks
            combined_text = self._combine_chunks(chunks)
            
            # Generate answer using AI or fallback method
            if self.openai_client:
                answer = self._generate_ai_answer(query, combined_text, doc_metadata['document_name'])
            else:
                answer = self._generate_fallback_answer(query, combined_text)
            
            # Create citations
            citations = self._create_citations(chunks)
            
            # Calculate overall relevance score
            relevance_score = sum(chunk['similarity_score'] for chunk in chunks) / len(chunks)
            
            return {
                'document_name': doc_metadata['document_name'],
                'doc_id': doc_metadata['doc_id'],
                'document_type': doc_metadata['document_type'],
                'answer': answer,
                'relevance_score': relevance_score,
                'citations': citations,
                'chunk_count': len(chunks),
                'pages_referenced': list(set(chunk['metadata']['page_number'] for chunk in chunks))
            }
            
        except Exception as e:
            print(f"Error generating document response: {str(e)}")
            return None
    
    def _combine_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Combine text chunks into coherent text
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Combined text
        """
        # Sort chunks by page and paragraph number
        sorted_chunks = sorted(chunks, key=lambda x: (
            x['metadata']['page_number'],
            x['metadata'].get('paragraph_number', 0)
        ))
        
        combined_text = ""
        for chunk in sorted_chunks:
            combined_text += chunk['text'] + "\n\n"
        
        return combined_text.strip()
    
    def _generate_ai_answer(self, query: str, context: str, doc_name: str) -> str:
        """
        Generate answer using OpenAI GPT
        
        Args:
            query: User query
            context: Relevant context from document
            doc_name: Document name
            
        Returns:
            Generated answer
        """
        try:
            prompt = f"""
            Based on the following text from the document "{doc_name}", please answer the user's question.
            
            User Question: {query}
            
            Document Context:
            {context}
            
            Please provide a comprehensive answer based solely on the information provided in the document context. 
            If the document doesn't contain enough information to fully answer the question, please indicate what 
            information is available and what might be missing.
            
            Answer:
            """
            
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context. Always base your answers on the given context and be clear about any limitations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating AI answer: {str(e)}")
            return self._generate_fallback_answer(query, context)
    
    def _generate_fallback_answer(self, query: str, context: str) -> str:
        """
        Generate answer using rule-based approach
        
        Args:
            query: User query
            context: Document context
            