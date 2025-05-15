# Use a lightweight Python image
FROM python:3.10-slim

# Set environment variables to prevent .pyc clutter
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system tools and gdown for Google Drive file download
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    && pip install --upgrade pip \
    && pip install gdown \
    && rm -rf /var/lib/apt/lists/*

# Copy your local app code
COPY . .

# Download large model files from Google Drive using gdown
# Word2Vec model
RUN gdown --id 12hsFkFMarwXpaQsAzHTkxEySZ6g020My -O word2vec_model.model
# Co-occurrence map
RUN gdown --id 19LUbJmhcPB_OHNYsIHwTf-ROqTfZzTAp -O co_occurrence_map.pkl

# Install Python packages
RUN pip install -r requirements.txt

# Expose Gradio’s default port
EXPOSE 7860

# Run your Gradio app
CMD ["python", "app.py"]
