# Automate testing etc

NOSETESTS ?= nosetests


all: install test

install:
	pip install -e .

test-coverage:
	nosetests -v --exe --doctest-tests --with-coverage --cover-package=tensorly tensorly

test:
	nosetests -v --exe --doctest-tests tensorly
