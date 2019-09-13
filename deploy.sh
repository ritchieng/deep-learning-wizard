#!/usr/bin/env bash

eval "$(conda shell.bash hook)"

echo Creating deployment folder
cd ..
sudo -u ritchie cp -rf docs-prog docs-prog-deploy
wait

echo Cleaning files for deployment...
cd docs-prog-deploy
sudo -u ritchie find . -name "*.pt" -exec rm {} \;
sudo -u ritchie find . -name "*-ubyte" -exec rm {} \;
sudo -u ritchie find . -name "*.ipynb" -exec rm {} \;
sudo -u ritchie find . -name "*.ipynb_checkpoints" -exec rmdir {} \;
sudo -u ritchie rm -rf ./docs/programming/electron/app-win32-x64/
sudo -u ritchie rm -rf ./docs/programming/electron/app/node_modules/
wait

echo Deploy to pages branch
mkdocs gh-deploy --force

echo Removing files
cd ..
sudo -u ritchie  rm -rf docs-prog-deploy

echo Reverting to original directory
cd docs-prog

echo Emptied thrash
emptytrash