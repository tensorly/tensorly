# Automate testing etc
BACKEND?='numpy'

.PHONY: all install debug test test-all test-coverage

all: install test

install:
	pip install -e .

debug:
	TENSORLY_BACKEND=$(BACKEND) pytest -v --pdb tensorly

test:
	TENSORLY_BACKEND=$(BACKEND) pytest -v tensorly

test-all:
	TENSORLY_BACKEND='numpy' pytest -v tensorly
	TENSORLY_BACKEND='cupy' pytest -v tensorly
	TENSORLY_BACKEND='pytorch' pytest -v tensorly
	TENSORLY_BACKEND='jax' pytest -v tensorly
	TENSORLY_BACKEND='tensorflow' pytest -v tensorly

test-coverage:
	TENSORLY_BACKEND=$(BACKEND) pytest -v --cov tensorly tensorly

