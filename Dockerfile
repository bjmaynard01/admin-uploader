FROM python:3.12-slim

# Dependencies
RUN apt-get update && apt-get install -y\
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    python3-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /app

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create upload folder
RUN mkdir -p uploads

# Flask Defaults
ENV FLASK_APP=main.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1
CMD ["flask", "run", "--host=0.0.0.0"]

