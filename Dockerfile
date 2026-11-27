# Reproducible environment for the hallucination experiment.
#
#   docker build -t llm-hallucination .
#   docker run --rm llm-hallucination                      # recompute every metric (no API keys needed)
#   docker run --rm llm-hallucination python -m pytest -q  # run the test suite
#
# To also run the open-source Hugging Face model (adds CPU-only PyTorch, ~1 GB):
#   docker build --build-arg WITH_OPEN_SOURCE=1 -t llm-hallucination:open-source .
#
# See docs/SETUP.md for running the full API experiment inside the container.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached when only code or data change.
COPY requirements.txt requirements-dev.txt requirements-open-source.txt ./
RUN pip install -r requirements-dev.txt

ARG WITH_OPEN_SOURCE=0
RUN if [ "$WITH_OPEN_SOURCE" = "1" ]; then \
        pip install torch --index-url https://download.pytorch.org/whl/cpu && \
        pip install -r requirements-open-source.txt ; \
    fi

COPY . .

# Default: reproduce the published metrics from the saved data.
CMD ["python", "src/analyze_results.py"]
