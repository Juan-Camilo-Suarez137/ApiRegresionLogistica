FROM python:3.11-slim-bookworm

WORKDIR /app

# Instalar dependencias Python directamente (sin paquetes del sistema)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la API
COPY src/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]