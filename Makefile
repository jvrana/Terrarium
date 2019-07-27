make:
	poetry self:update
	poetry install
	poetry run pre-commit install


format:
	poetry run black terrarium tests


test:
	poetry run pytest


clean:
	rm -rf tests/.pytest_cache
	rm -rf tests/live_tests/fixtures
	rm -rf .pytest_cache
	rm -rf pip-wheel-*


benchmark:
	rm -rf .benchmarks/images/*svg
	poetry run python -m pytest -m benchmark --benchmark-autosave --benchmark-compare=0001 --benchmark-histogram=assets/benchmark/histogram