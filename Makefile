
init:
	pip install pip -U
	curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
	poetry self:update
	poetry install
	poetry run pre-commit install

docs:
	echo "No documentation"

format:
	poetry run black terrarium tests

release:
	sh scripts/interactive_release.sh