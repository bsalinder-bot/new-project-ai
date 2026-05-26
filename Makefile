.PHONY: install run test docker-build docker-run

install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

run:
	python app.py

test:
	python -m unittest discover -s tests

docker-build:
	docker build -t ghcr.io/bsalinder-bot/new-project-ai:latest .

docker-run:
	docker run --rm -p 5000:5000 ghcr.io/bsalinder-bot/new-project-ai:latest

compose-up:
	docker-compose up --build

compose-down:
	docker-compose down

k8s-apply:
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml

docker-push:
	@echo "Requires DOCKER_REGISTRY and DOCKER_IMAGE environment variables."
	@echo "Example: DOCKER_REGISTRY=ghcr.io/owner DOCKER_IMAGE=new-project-ai make docker-push"
	docker build -t ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest .
	docker push ${DOCKER_REGISTRY}/${DOCKER_IMAGE}:latest

cli:
	python cli.py

k8s-apply:
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
