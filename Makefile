.PHONY: install run test samples docker

install:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:app --reload

test:
	DETECTOR=stub pytest -q

samples:
	python -m scripts.make_samples

docker:
	docker build -t taglens . && docker run --rm -p 8000:8000 taglens
