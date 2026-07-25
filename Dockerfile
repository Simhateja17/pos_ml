FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md job.py forecast.py eligibility.py writeback.py ./
RUN pip install --no-cache-dir .
CMD ["couture-forecast"]
