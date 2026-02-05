FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Keep container alive so docker exec can be used
CMD ["bash", "-lc", "tail -f /dev/null"]