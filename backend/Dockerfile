FROM python:3.10

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r backend/requirements.txt

CMD ["streamlit", "run", "backend/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
