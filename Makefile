.PHONY: install format lint test clean run-api

install:
	pip install -r requirements.txt

format:
	black src tests
	isort src tests

lint:
	flake8 src tests
	mypy src

test:
	PYTHONPATH=. pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-api:
	uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
