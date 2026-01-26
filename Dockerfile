FROM python:3.9-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
ENV PYTHONPATH=/app/src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "src/model_training.py"]
CMD ["LSTM"]