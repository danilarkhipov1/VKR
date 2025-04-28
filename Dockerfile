FROM python:3.8-slim

WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY ./src /app/src
COPY ./models /app/models

# Устанавливаем переменные окружения
ENV MODEL_PATH=/app/models/model.pt
ENV PYTHONPATH=/app

# Запускаем приложение
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]