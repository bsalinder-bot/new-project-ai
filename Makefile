.PHONY: install run test docker-build docker-run

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

run:
	python app.py

test:
	python -m unittest discover -s tests

docker-build:
	docker build -t planetary-translator:latest .

docker-run:
	docker run --rm -p 5000:5000 planetary-translator:latest

docker-push:
	@echo "Requires DOCKER_REGISTRY and DOCKER_IMAGE environment variables."
	@echo "Example: DOCKER_REGISTRY=ghcr.io/owner DOCKER_IMAGE=new-project-ai make docker-push"
	docker build -t ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest .
	docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest

cli:
	python cli.py
