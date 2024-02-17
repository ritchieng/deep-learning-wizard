# !/usr/bin/env bash

eval "$(conda shell.bash hook)"

echo Creating deployment folder...
cd ..
sudo -u $(whoami) cp -rf deep-learning-wizard dlw-deploy
wait

echo Cleaning files for deployment...
cd dlw-deploy
sudo find . -name "*.simg" -exec rm {} \;
sudo find . -name "*.pt" -exec rm {} \;
sudo find . -name "*-ubyte" -exec rm {} \;
sudo find . -name "*.ipynb" -exec rm {} \;
sudo find . -name "*.ipynb_checkpoints" -exec rmdir {} \;
sudo rm -rf ./docs/programming/electron/app-win32-x64/
sudo rm -rf ./docs/programming/electron/app/node_modules/
wait

echo Deploying to pages branch...
mkdocs gh-deploy --force

echo Removing files...
cd ..
sudo rm -rf dlw-deploy

echo Reverting to original directory...
cd deep-learning-wizard

echo Emptying thrash
trash-empty

echo Deployed site!