FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY main.py .
COPY saved_models/ saved_models/

# Expose port
EXPOSE 8501

# Run app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]