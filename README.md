# Document Research & Theme Identification Chatbot

This project allows users to upload various documents (PDF, DOCX, text, images), ask natural language questions, and receive summarized answers with detected common themes across documents.

## Features

- Document upload and OCR support
- Embedding-based vector search using ChromaDB
- GPT-based or fallback QA and theme detection
- Streamlit UI interface

## Setup

```bash
pip install -r requirements.txt
streamlit run backend/app/main.py
```

## Project Structure

```
chatbot_theme_identifier/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   ├── services/
│   │   ├── main.py
│   │   └── config.py
│   ├── data/
│   ├── Dockerfile
│   └── requirements.txt
├── docs/
├── tests/
├── demo/
└── README.md
```
