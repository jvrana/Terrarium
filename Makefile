
init:
	curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
	poetry self:update
	poetry install
	poetry run pre-commit install

docs:
    echo "No documentation"

format:
	poetry run black terrarium tests

release:
    sh scripts/quick_release.sh