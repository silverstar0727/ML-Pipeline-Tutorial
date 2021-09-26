FROM us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-6:latest

WORKDIR /app

# Installs hypertune library
RUN pip install cloudml-hypertune

# Copies the trainer code to the docker image.
COPY . .

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "task.py"]
