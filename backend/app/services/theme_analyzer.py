import openai
import os
from typing import List, Dict, Any, Optional
import json
from collections import Counter, defaultdict
import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class ThemeAnalyzer:
    """
    Analyzes document responses to identify common themes and patterns
    """
    
    def __init__(self):
        # Initialize OpenAI client (you'll need to set your API key)
        self.openai_client = None
        self._initialize_openai()
        
        # Theme analysis settings
        self.min_theme_support = 2  # Minimum documents supporting a theme
        self.max_themes = 5  # Maximum themes to identify
        
        # TF-IDF vectorizer for text analysis
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.95,
            min_df=0.02
        )
    
    def _initialize_openai(self):
        """Initialize OpenAI client if API key is available"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                openai.api_key = api_key
                self.openai_client = openai
                print("OpenAI client initialized successfully")
            else:
                print("OpenAI API key not found. Using fallback theme analysis methods.")
        except Exception as e:
            print(f"Error initializing OpenAI: {str(e)}")
            self.openai_client = None
    
    def analyze_themes(self, query: str, document_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze document responses to identify common themes
        
        Args:
            query: Original user query
            document_responses: List of responses from individual documents
            
        Returns:
            List of identified themes with summaries and supporting documents
        """
        try:
            if len(document_responses) < 2:
                return self._single_document_theme(query, document_responses)
            
            # Try AI-powered theme analysis first
            if self.openai_client:
                themes = self._ai_theme_analysis(query, document_responses)
                if themes:
                    return themes
            
            # Fallback to rule-based theme analysis
            return self._rule_based_theme_analysis(query, document_responses)
            
        except Exception as e:
            print(f"Error in theme analysis: {str(e)}")
            return self._fallback_theme_analysis(query, document_responses)
    
    def _ai_theme_analysis(self, query: str, document_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Use AI (OpenAI GPT) to identify themes
        
        Args:
            query: User query
            document_responses: Document responses
            
        Returns:
            List of identified themes
        """
        try:
            # Prepare context for AI analysis
            context = self._prepare_ai_context(query, document_responses)
            
            # Create prompt for theme identification
            prompt = f"""
            Analyze the following document responses to identify common themes and patterns.
            
            Original Query: {query}
            
            Document Responses:
            {context}
            
            Please identify 2-5 main themes across these responses. For each theme, provide:
            1. A clear title (2-5 words)
            2. A comprehensive summary (2-3 sentences)
            3. List of supporting documents by name
            4. Key evidence or quotes that support this theme
            
            Format your response as JSON with this structure:
            {{
                "themes": [
                    {{
                        "title": "Theme Title",
                        "summary": "Detailed summary of the theme",
                        "supporting_documents": ["Document1.pdf", "Document2.pdf"],
                        "key_evidence": ["Evidence 1", "Evidence 2"],
                        "confidence_score": 0.85
                    }}
                ]
            }}
            """
            
            # Call OpenAI API
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert analyst specializing in identifying themes and patterns across multiple documents."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3
            )
            
            # Parse response
            ai_response = response.choices[0].message.content
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                themes_data = json.loads(json_match.group())
                return themes_data.get('themes', [])
            
            return []
            
        except Exception as e:
            print(f"Error in AI theme analysis: {str(e)}")
            return None
    
    def _rule_based_theme_analysis(self, query: str, document_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rule-based theme identification using NLP techniques
        
        Args:
            query: User query
            document_responses: Document responses
            
        Returns:
            List of identified themes
        """
        try:
            # Extract texts for analysis
            texts = [resp['answer'] for resp in document_responses]
            
            # Perform TF-IDF analysis
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Cluster similar responses
            n_clusters = min(3, len(document_responses))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(tfidf_matrix)
            else:
                clusters = [0] * len(document_responses)
            
            # Analyze each cluster
            themes = []
            for cluster_id in set(clusters):
                cluster_docs = [i for i, c in enumerate(clusters) if c == cluster_id]
                
                if len(cluster_docs) >= self.min_theme_support:
                    theme = self._analyze_cluster_theme(
                        cluster_id, cluster_docs, document_responses, tfidf_matrix, feature_names
                    )
                    if theme:
                        themes.append(theme)
            
            # If no clusters found, create themes based on keyword similarity
            if not themes:
                themes = self._keyword_based_themes(query, document_responses)
            
            return themes[:self.max_themes]
            
        except Exception as e:
            print(f"Error in rule-based theme analysis: {str(e)}")
            return self._fallback_theme_analysis(query, document_responses)
    
    def _analyze_cluster_theme(self, cluster_id: int, cluster_docs: List[int], 
                              document_responses: List[Dict[str, Any]], 
                              tfidf_matrix: Any, feature_names: List[str]) -> Dict[str, Any]:
        """
        Analyze a cluster of documents to identify theme
        
        Args:
            cluster_id: Cluster identifier
            cluster_docs: Document indices in cluster
            document_responses: All document responses
            tfidf_matrix: TF-IDF matrix
            feature_names: Feature names from TF-IDF
            
        Returns:
            Theme dictionary
        """
        try:
            # Get cluster documents
            cluster_responses = [document_responses[i] for i in cluster_docs]
            
            # Calculate average TF-IDF scores for cluster
            cluster_tfidf = tfidf_matrix[cluster_docs].mean(axis=0).A1
            
            # Get top keywords
            top_indices = cluster_tfidf.argsort()[-10:][::-1]
            top_keywords = [feature_names[i] for i in top_indices if cluster_tfidf[i] > 0]
            
            # Create theme title from top keywords
            title = self._generate_theme_title(top_keywords)
            
            # Create summary
            summary = self._generate_theme_summary(cluster_responses, top_keywords)
            
            # Get supporting documents
            supporting_docs = [resp['document_name'] for resp in cluster_responses]
            
            # Calculate confidence score
            confidence = self._calculate_theme_confidence(cluster_responses, top_keywords)
            
            return {
                'title': title,
                'summary': summary,
                'supporting_documents': supporting_docs,
                'key_keywords': top_keywords[:5],
                'confidence_score': confidence,
                'document_count': len(cluster_responses)
            }
            
        except Exception as e:
            print(f"Error analyzing cluster theme: {str(e)}")
            return None
    
    def _keyword_based_themes(self, query: str, document_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create themes based on keyword frequency and similarity
        
        Args:
            query: User query
            document_responses: Document responses
            
        Returns:
            List of themes
        """
        try:
            # Extract keywords from all responses
            all_keywords = []
            doc_keywords = {}
            
            for i, resp in enumerate(document_responses):
                keywords = self._extract_keywords(resp['answer'])
                doc_keywords[i] = keywords
                all_keywords.extend(keywords)
            
            # Find most common keywords
            keyword_counts = Counter(all_keywords)
            common_keywords = [k for k, v in keyword_counts.most_common(20) if v >= 2]
            
            # Group documents by shared keywords
            themes = []
            used_docs = set()
            
            for keyword in common_keywords:
                supporting_docs = []
                for doc_idx, keywords in doc_keywords.items():
                    if keyword in keywords and doc_idx not in used_docs:
                        supporting_docs.append(doc_idx)
                
                if len(supporting_docs) >= self.min_theme_support:
                    theme_responses = [document_responses[i] for i in supporting_docs]
                    
                    theme = {
                        'title': f"Theme: {keyword.title()}",
                        'summary': self._generate_keyword_theme_summary(keyword, theme_responses),
                        'supporting_documents': [resp['document_name'] for resp in theme_responses],
                        'key_keywords': [keyword],
                        'confidence_score': min(0.8, len(supporting_docs) / len(document_responses)),
                        'document_count': len(supporting_docs)
                    }
                    
                    themes.append(theme)
                    used_docs.update(supporting_docs)
                    
                    if len(themes) >= self.max_themes:
                        break
            
            return themes
            
        except Exception as e:
            print(f"Error in keyword-based theme analysis: {str(e)}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common words
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'her', 'way', 'many', 'then', 'them', 'these', 'so', 'some', 'would', 'make', 'like', 'into', 'him', 'time', 'has', 'look', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'oil', 'sit', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part'}
        
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        
        # Return most frequent keywords
        return list(Counter(keywords).most_common(10))
    
    def _generate_theme_title(self, keywords: List[str]) -> str:
        """
        Generate a theme title from keywords
        
        Args:
            keywords: List of keywords
            
        Returns:
            Theme title
        """
        if not keywords:
            return "General Theme"
        
        # Take top 2-3 keywords and create a title
        title_words = keywords[:3]
        return " & ".join([word.title() for word in title_words])
    
    def _generate_theme_summary(self, responses: List[Dict[str, Any]], keywords: List[str]) -> str:
        """
        Generate a theme summary
        
        Args:
            responses: Document responses in theme
            keywords: Theme keywords
            
        Returns:
            Theme summary
        """
        try:
            # Extract key sentences containing keywords
            key_sentences = []
            
            for resp in responses:
                sentences = resp['answer'].split('.')
                for sentence in sentences:
                    if any(keyword.lower() in sentence.lower() for keyword in keywords[:3]):
                        key_sentences.append(sentence.strip())
            
            # Create summary
            if key_sentences:
                summary = f"This theme appears in {len(responses)} documents and focuses on {', '.join(keywords[:3])}. "
                summary += f"Key insights include: {'. '.join(key_sentences[:2])}."
            else:
                summary = f"This theme appears across {len(responses)} documents and relates to {', '.join(keywords[:3])}."
            
            return summary
            
        except Exception as e:
            print(f"Error generating theme summary: {str(e)}")
            return f"Theme identified across {len(responses)} documents."
    
    def _generate_keyword_theme_summary(self, keyword: str, responses: List[Dict[str, Any]]) -> str:
        """
        Generate summary for keyword-based theme
        
        Args:
            keyword: Central keyword
            responses: Supporting responses
            
        Returns:
            Theme summary
        """
        return f"This theme centers around '{keyword}' and appears in {len(responses)} documents. The documents provide various perspectives and information related to this topic."
    
    def _calculate_theme_confidence(self, responses: List[Dict[str, Any]], keywords: List[str]) -> float:
        """
        Calculate confidence score for a theme
        
        Args:
            responses: Theme responses
            keywords: Theme keywords
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # Base confidence on number of supporting documents
            doc_score = min(1.0, len(responses) / 3)
            
            # Boost confidence if keywords appear frequently
            keyword_mentions = 0
            total_words = 0
            
            for resp in responses:
                words = resp['answer'].lower().split()
                total_words += len(words)
                for keyword in keywords[:3]:
                    keyword_mentions += words.count(keyword.lower())
            
            keyword_score = min(1.0, keyword_mentions / max(total_words, 1) * 100)
            
            # Combine scores
            confidence = (doc_score + keyword_score) / 2
            
            return round(confidence, 2)
            
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _single_document_theme(self, query: str, document_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Handle case with single document
        
        Args:
            query: User query
            document_responses: Single document response
            
        Returns:
            Single theme
        """
        if not document_responses:
            return []
        
        resp = document_responses[0]
        keywords = self._extract_keywords(resp['answer'])
        
        return [{
            'title': f"Primary Topic: {query[:30]}",
            'summary': f"Based on the single document analysis, the main topic relates to the query about {query}. {resp['answer'][:200]}...",
            'supporting_documents': [resp['document_name']],
            'key_keywords': [k[0] for k in keywords[:5]],
            'confidence_score': 0.7,
            'document_count': 1
        }]
    
    def _fallback_theme_analysis(self, query: str, document_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback theme analysis when other methods fail
        
        Args:
            query: User query
            document_responses: Document responses
            
        Returns:
            Basic themes
        """
        themes = []
        
        # Group by document type or create single theme
        if len(document_responses) > 1:
            theme = {
                'title': f"General Findings: {query[:30]}",
                'summary': f"Analysis of {len(document_responses)} documents reveals various perspectives on the query: {query}. The documents provide complementary information and insights.",
                'supporting_documents': [resp['document_name'] for resp in document_responses],
                'key_keywords': [],
                'confidence_score': 0.6,
                'document_count': len(document_responses)
            }
            themes.append(theme)
        
        return themes
    
    def _prepare_ai_context(self, query: str, document_responses: List[Dict[str, Any]]) -> str:
        """
        Prepare context for AI analysis
        
        Args:
            query: User query
            document_responses: Document responses
            
        Returns:
            Formatted context string
        """
        context = ""
        for i, resp in enumerate(document_responses, 1):
            context += f"\nDocument {i}: {resp['document_name']}\n"
            context += f"Response: {resp['answer'][:500]}...\n"
            context += f"Relevance Score: {resp['relevance_score']}\n"
            context += "---\n"
        
        return context