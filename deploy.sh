#!/usr/bin/env bash
echo Cleaning files for deploy...
sudo find . -name "*.pt" -exec rm {} \;
sudo find . -name "*-ubyte" -exec rm {} \;
sudo find . -name "*.ipynb" -exec rm {} \;
sudo find . -name "*.ipynb_checkpoints" -exec rmdir {} \;
sudo rm -rf ./docs/programming/electron/app-win32-x64/
sudo rm -rf ./docs/programming/electron/app/node_modules/

echo Deploy to pages branch
mkdocs gh-deploy --force

echo Reset file deletes
git stash