FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Install dependencies
ENV tz=Etc/UTC DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
  git \
  libgl1 \
  libglib2.0-0

WORKDIR /app

COPY models ./models

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

COPY src .

CMD ["python", "server.py"]