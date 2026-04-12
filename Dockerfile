FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY model.pkl .

EXPOSE 5000
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]