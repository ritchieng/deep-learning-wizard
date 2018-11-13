#!/usr/bin/env bash
mkdocs build
echo Completed building site!

cd site
echo Cleaning crap files...
find . -name "*.pt" -exec rm {} \;
find . -name "*-ubyte" -exec rm {} \;
find . -name "*.ipynb" -exec rm {} \;
find . -name "*.ipynb_checkpoints" -exec rm {} \;
cd ..
echo Removed all files

echo copying site for backup
cp -rf ./site ../site

echo Pushing files to github
git checkout gh-pages
#rm -rf *
#
#cp -rf ../site ./
#touch .nojekyll
#cd ..
#git checkout gh-pages
#git add ./site
#git commit -m "Manual deploy"
#git push origin gh-pages --force

echo Checkout master
git checkout master

