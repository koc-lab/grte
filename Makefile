EXECUTER:=poetry run

all: format lint requirements

install:
	git init
	$(EXECUTER) poetry install
	$(EXECUTER) pre-commit install

clean:
	rm -rf .mypy_cache .pytest_cache .coverage htmlcov
	$(EXECUTER) ruff clean

requirements:
	poetry export -f requirements.txt -o requirements.txt --with dev --without-hashes

format:
	$(EXECUTER) ruff format .

lint:
	$(EXECUTER) ruff check . --fix
	$(EXECUTER) mypy .
