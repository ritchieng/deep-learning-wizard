# !/usr/bin/env bash

eval "$(conda shell.bash hook)"

echo Creating deployment folder...
cd ..
sudo -u $(whoami) cp -rf deep-learning-wizard dlw-deploy
wait

echo Cleaning files for deployment...
cd dlw-deploy
find . -name "*.simg" -exec rm {} \;
echo Cleaned *.simg
find . -name "*.pt" -exec rm {} \;
echo Cleaned *.pt
find . -name "*-ubyte" -exec rm {} \;
echo Cleaned *.-ubyte
find . -name "*.ipynb" -exec rm {} \;
find . -name "*.ipynb_checkpoints" -exec rmdir {} \;
echo Cleaned *.ipynb
rm -rf ./docs/programming/electron/app-win32-x64/
rm -rf ./docs/programming/electron/app/node_modules/
echo Cleaned electron
wait

echo Deploying to pages branch...
mkdocs gh-deploy --force

echo Removing files...
cd ..
rm -rf dlw-deploy

echo Reverting to original directory...
cd deep-learning-wizard

echo Emptying thrash
trash-empty -f

echo Deployed site!