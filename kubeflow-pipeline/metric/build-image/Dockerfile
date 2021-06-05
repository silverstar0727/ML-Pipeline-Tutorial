FROM tensorflow/tensorflow:2.1.0-py3

RUN mkdir -p /app
ADD mnist_simple.py /app/
WORKDIR /app

CMD ["python", "/app/mnist_simple.py"]