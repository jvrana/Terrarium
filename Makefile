
init:
	curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
	poetry self:update
	poerty install
	poetry run pre-commit install


format:
	poetry run black terrarium tests