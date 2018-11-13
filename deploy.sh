#!/usr/bin/env bash
echo Cleaning files for deploy...
find . -name "*.pt" -exec rm {} \;
find . -name "*-ubyte" -exec rm {} \;
find . -name "*.ipynb" -exec rm {} \;
find . -name "*.ipynb_checkpoints" -exec rm {} \;

echo Deploy to pages branch
mkdocs gh-deploy --force

echo Reset file deletes
git stash