#!/usr/bin/env bash
mkdocs build
echo Completed building site!

cd site
echo Cleaning crap files...
find . -name "*.pt" -exec rm {} \;
find . -name "*-ubyte" -exec rm {} \;
find . -name "*.ipynb" -exec rm {} \;
find . -name "*.ipynb_checkpoints" -exec rm {} \;
echo Removed all files

echo Pushing files to github
cd ..
git checkout gh-pages
cd site
touch .nojekyll
cd ..
git checkout gh-pages
git add ./site
git commit -m "Manual deploy"
git push origin gh-pages --force

echo Checkout master
git checkout master

