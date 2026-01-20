FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY reelmotion_mcp/ ./reelmotion_mcp/

# Crear directorio para archivos temporales
RUN mkdir -p temp_files

# Cloud Run uses PORT env variable (default 8080)
EXPOSE 8080

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080
ENV HOST=0.0.0.0

# Use exec form - PORT is read from env inside the script
CMD ["python", "reelmotion_mcp/server.py", "http"]
