#!/bin/bash

# Inspired from scikit-image
# https://github.com/scikit-image/scikit-image/blob/master/tools/travis/deploy_docs.sh
(
    git config --global user.email "admin@tensorly.org"
    git config --global user.name "Travis"

	pip install slimit rcssmin matplotlib sphinx
	cd doc
	make html
	cd ..

    git clone --quiet https://github.com/tensorly/tensorly.github.io doc_folder
    cd doc_folder

    git rm -r dev/*
    cp -r ../doc/_build/html dev/
    git add dev

    git commit -m "Auto-update"
    git push --force --quiet "https://${gh_token}@github.com/tensorly/tensorly.github.io" > /dev/null 2>&1
)
