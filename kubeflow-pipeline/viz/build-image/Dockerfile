FROM python:3.6.10-slim

RUN pip install sklearn pandas

COPY ./ /app
WORKDIR /app

CMD ["python", "/app/confusion_matrix.py"]