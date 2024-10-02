# Requirements stage
FROM python:3.10-slim as requirements
WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN pip install poetry
# Export Poetry dependencies to requirements.txt
RUN poetry export -f requirements.txt --without-hashes -o ./requirements.txt

# Final stage
FROM python:3.10-slim
WORKDIR /app

ENV PYTHONPATH="${PYTHONPATH}:/app/"

COPY . /app
COPY --from=requirements /app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt


CMD ["python", "video_analysis/main.py"]
