.PHONY: install run test

install:
	python -m pip install -r requirements.txt

run:
	python app.py

test:
	python -m unittest discover -s tests
