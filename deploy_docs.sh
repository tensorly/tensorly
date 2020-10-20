#!/bin/bash
# Inspired from scikit-image
# https://github.com/scikit-image/scikit-image/blob/master/tools/travis/deploy_docs.sh
if [[ $TRAVIS_PULL_REQUEST == false && $TRAVIS_BRANCH == "master" &&
	      $DEPLOY_DOCS == 1 ]]
then
	echo "---------------------------------------------------"
	echo "deploy_docs.sh: building doc and pushing to website"
	(
    git config --global user.email "admin@tensorly.org"
    git config --global user.name "Travis"

	# Install the dependencies
	cd doc
	echo "-- Installing dependencies"
	conda install matplotlib Pillow sphinx
	pip install -r requirements_doc.txt

	# Making the doc
	echo "-- Building doc"
	make html

	# If build succeeded, update website
	if [ $? -eq 0 ]; then
		echo 'Documentation was successfully built, updating the website.'
		cd ..

		echo "-- Cloning doc repo"
		git clone --quiet https://github.com/tensorly/tensorly.github.io doc_folder
		cd doc_folder
		git rm -r dev/*
		cp -r ../doc/_build/html/* dev/

		echo "Pushinng to git"
		git add dev
		git commit -m "Travis auto-update"
		git push --force --quiet "https://${gh_token}@github.com/tensorly/tensorly.github.io" > /dev/null 2>&1
	# Build failed
	else
		echo '-- Build FAILED. Not updating the website.'
	fi
	)
else
    echo "NOT BUILDIGN DOC -- will only push docs from master and if DEPLOY_DOCS == 1"
fi

