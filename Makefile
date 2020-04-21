# Automate testing etc
BACKEND?='numpy'

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
	TENSORLY_BACKEND='mxnet' pytest -v tensorly
	TENSORLY_BACKEND='jax' pytest -v tensorly
	TENSORLY_BACKEND='tensorflow' pytest -v tensorly

test-coverage:
	TENSORLY_BACKEND=$(BACKEND) pytest -v --cov tensorly tensorly

