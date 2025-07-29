# Use official slim Python base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-tha \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libgl1 \
    fonts-thai-tlwg \
    fonts-dejavu-core \
    unzip \
    wget \
    locales && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files first
COPY . .

# Download and extract EasyOCR model zip files
RUN mkdir -p /app/models/.EasyOCR/recognition && \
    wget -O english_g2.zip https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip && \
    wget -O thai.zip https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/thai.zip && \
    wget -O craft_mlt_25k.zip https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip && \
    unzip english_g2.zip -d /app/models/.EasyOCR/ && \
    unzip thai.zip -d /app/models/.EasyOCR/ && \
    unzip craft_mlt_25k.zip -d /app/models/.EasyOCR/ && \
    rm *.zip

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 10000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
