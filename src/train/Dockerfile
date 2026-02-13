FROM python:3.11-slim

WORKDIR /app
# Installing deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY src /app/src
#"src" importable
ENV PYTHONPATH=/app
# running training
ENTRYPOINT ["python", "-m", "src.train.train"]