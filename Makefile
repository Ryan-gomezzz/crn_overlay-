install:
	pip install -r requirements.txt
	pre-commit install

test:
	pytest tests/

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check --fix .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
