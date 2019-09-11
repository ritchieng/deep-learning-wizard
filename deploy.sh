#!/usr/bin/env bash

echo Creating deployment folder
cd ..
sudo cp -rf docs-prog docs-prog-deploy
cd docs-prog-deploy

echo Cleaning files for deployment...
sudo find . -name "*.pt" -exec rm {} \;
sudo find . -name "*-ubyte" -exec rm {} \;
sudo find . -name "*.ipynb" -exec rm {} \;
sudo find . -name "*.ipynb_checkpoints" -exec rmdir {} \;
sudo rm -rf ./docs/programming/electron/app-win32-x64/
sudo rm -rf ./docs/programming/electron/app/node_modules/

echo Deploy to pages branch
mkdocs gh-deploy --force

echo Removing files
cd ..
sudo rm -rf docs-prog-deploy

echo Reverting to original directory
cd docs-prog