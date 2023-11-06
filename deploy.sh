# !/usr/bin/env bash

eval "$(conda shell.bash hook)"

echo Creating deployment folder...
cd ..
sudo -u ritchieng cp -rf deep-learning-wizard dlw-deploy
wait

echo Cleaning files for deployment...
cd dlw-deploy
sudo find . -name "*.simg" -exec rm {} \;
echo Done S1
sudo find . -name "*.pt" -exec rm {} \;
echo Done S2
sudo find . -name "*-ubyte" -exec rm {} \;
echo Done S3
sudo find . -name "*.ipynb" -exec rm {} \;
echo Done S4
sudo find . -name "*.ipynb_checkpoints" -exec rmdir {} \;
echo Done S5
sudo rm -rf ./docs/programming/electron/app-win32-x64/
echo Done S6
sudo rm -rf ./docs/programming/electron/app/node_modules/
echo Done S7
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