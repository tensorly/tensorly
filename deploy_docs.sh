#!/bin/bash
# Inspired from scikit-image
# https://github.com/scikit-image/scikit-image/blob/master/tools/travis/deploy_docs.sh
if [[ $TRAVIS_PULL_REQUEST == false && $TRAVIS_BRANCH == "master" &&
	      $DEPLOY_DOCS == 1 ]]
then
	echo "-- pushing docs --"
	(
    git config --global user.email "admin@tensorly.org"
    git config --global user.name "Travis"

	# Installed the dependencies
	conda install matplotlib Pillow sphinx
	pip install slimit rcssmin 

	# cd to the doc folder and build the doc
	cd doc
	make html
	cd ..

    git clone --quiet https://github.com/tensorly/tensorly.github.io doc_folder
    cd doc_folder
    git rm -r dev/*
    cp -r ../doc/_build/html/* dev/

    git add dev
    git commit -m "Travis auto-update"
    git push --force --quiet "https://${gh_token}@github.com/tensorly/tensorly.github.io" > /dev/null 2>&1
	)
else
    echo "-- will only push docs from master and if DEPLOY_DOCS == 1--"
fi

