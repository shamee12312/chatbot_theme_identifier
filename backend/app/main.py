import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
from typing import List, Dict, Any
import json
from datetime import datetime

# Import our custom modules
from document_processor import DocumentProcessor
from vector_store import VectorStore
from theme_analyzer import ThemeAnalyzer
from query_engine import QueryEngine

# Configure Streamlit page
st.set_page_config(
    page_title="Document Research & Theme Identification Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class DocumentChatbot:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.theme_analyzer = ThemeAnalyzer()
        self.query_engine = QueryEngine(self.vector_store, self.theme_analyzer)
        
        # Initialize session state
        if 'documents' not in st.session_state:
            st.session_state.documents = []
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        if 'processed_docs' not in st.session_state:
            st.session_state.processed_docs = {}
    
    def render_sidebar(self):
        """Render the sidebar for document management"""
        st.sidebar.title("📚 Document Management")
        
        # File uploader
        uploaded_files = st.sidebar.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Upload PDFs, text files, images, or Word documents"
        )
        
        if uploaded_files:
            if st.sidebar.button("Process Documents"):
                self.process_uploaded_files(uploaded_files)
        
        # Display uploaded documents
        if st.session_state.documents:
            st.sidebar.subheader(f"📄 Documents ({len(st.session_state.documents)})")
            for i, doc in enumerate(st.session_state.documents):
                with st.sidebar.expander(f"{doc['name'][:30]}..."):
                    st.write(f"**Type:** {doc['type']}")
                    st.write(f"**Size:** {doc['size']} bytes")
                    st.write(f"**Pages:** {doc.get('pages', 'N/A')}")
                    if st.button(f"Remove", key=f"remove_{i}"):
                        self.remove_document(i)
                        st.rerun()
    
    def process_uploaded_files(self, files):
        """Process uploaded files and extract text"""
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        for i, file in enumerate(files):
            status_text.text(f"Processing {file.name}...")
            
            # Save file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.name).suffix) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Process document
                doc_data = self.doc_processor.process_document(tmp_path, file.name)
                if doc_data:
                    # Add to vector store
                    self.vector_store.add_document(doc_data)
                    
                    # Store in session state
                    st.session_state.documents.append({
                        'name': file.name,
                        'type': file.type,
                        'size': file.size,
                        'pages': doc_data.get('total_pages', 1),
                        'doc_id': doc_data['doc_id']
                    })
                    
                    st.session_state.processed_docs[doc_data['doc_id']] = doc_data
                
            except Exception as e:
                st.sidebar.error(f"Error processing {file.name}: {str(e)}")
            
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
            
            progress_bar.progress((i + 1) / len(files))
        
        status_text.text("✅ Processing complete!")
        st.sidebar.success(f"Successfully processed {len(files)} documents!")
    
    def remove_document(self, index):
        """Remove a document from the system"""
        if 0 <= index < len(st.session_state.documents):
            doc = st.session_state.documents.pop(index)
            doc_id = doc['doc_id']
            
            # Remove from vector store
            self.vector_store.remove_document(doc_id)
            
            # Remove from processed docs
            if doc_id in st.session_state.processed_docs:
                del st.session_state.processed_docs[doc_id]
    
    def render_main_interface(self):
        """Render the main chat interface"""
        st.title("🔍 Document Research & Theme Identification Chatbot")
        st.markdown("Ask questions about your uploaded documents and discover common themes!")
        
        # Check if documents are uploaded
        if not st.session_state.documents:
            st.warning("Please upload at least one document to get started.")
            return
        
        # Display document stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", len(st.session_state.documents))
        with col2:
            total_pages = sum(doc.get('pages', 1) for doc in st.session_state.documents)
            st.metric("Total Pages", total_pages)
        with col3:
            st.metric("Queries Asked", len(st.session_state.query_history))
        
        # Query input
        query = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g., What are the main findings about climate change?",
            key="query_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Search", type="primary")
        with col2:
            if st.session_state.query_history:
                if st.button("📋 Clear History"):
                    st.session_state.query_history = []
                    st.rerun()
        
        # Process query
        if search_button and query:
            self.process_query(query)
        
        # Display query history
        if st.session_state.query_history:
            st.subheader("📊 Query Results")
            for i, result in enumerate(reversed(st.session_state.query_history)):
                with st.expander(f"Query {len(st.session_state.query_history) - i}: {result['query'][:50]}..."):
                    self.display_query_result(result)
    
    def process_query(self, query: str):
        """Process a user query and generate results"""
        with st.spinner("Searching documents and analyzing themes..."):
            try:
                # Execute query
                result = self.query_engine.process_query(query)
                
                # Add timestamp
                result['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                result['query'] = query
                
                # Store in history
                st.session_state.query_history.append(result)
                
                st.success("Query processed successfully!")
                
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    def display_query_result(self, result: Dict[str, Any]):
        """Display query results with themes and citations"""
        st.write(f"**Query:** {result['query']}")
        st.write(f"**Time:** {result['timestamp']}")
        
        # Display individual document answers
        st.subheader("📄 Document Responses")
        
        if result.get('document_responses'):
            # Create DataFrame for better display
            df_data = []
            for doc_response in result['document_responses']:
                df_data.append({
                    'Document': doc_response['document_name'],
                    'Relevance Score': f"{doc_response['relevance_score']:.3f}",
                    'Answer': doc_response['answer'][:200] + "..." if len(doc_response['answer']) > 200 else doc_response['answer'],
                    'Citations': doc_response['citations']
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Detailed view toggle
            if st.checkbox("Show detailed responses", key=f"detail_{result['timestamp']}"):
                for doc_response in result['document_responses']:
                    with st.expander(f"📄 {doc_response['document_name']}"):
                        st.write(f"**Relevance Score:** {doc_response['relevance_score']:.3f}")
                        st.write(f"**Answer:** {doc_response['answer']}")
                        st.write(f"**Citations:** {doc_response['citations']}")
        
        # Display themes
        st.subheader("🎯 Identified Themes")
        
        if result.get('themes'):
            for i, theme in enumerate(result['themes'], 1):
                with st.container():
                    st.markdown(f"### Theme {i}: {theme['title']}")
                    st.write(theme['summary'])
                    
                    # Show supporting documents
                    if theme.get('supporting_documents'):
                        st.write("**Supporting Documents:**")
                        for doc in theme['supporting_documents']:
                            st.write(f"- {doc}")
                    
                    st.markdown("---")
        else:
            st.info("No specific themes identified for this query.")
    
    def run(self):
        """Main application runner"""
        # Render sidebar
        self.render_sidebar()
        
        # Render main interface
        self.render_main_interface()

# Run the application
if __name__ == "__main__":
    app = DocumentChatbot()
    app.run()