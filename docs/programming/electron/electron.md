---
comments: true
---

# Electron

## Why Electron

We choose to cover electron as you can easily use it as a front-end application across any platforms (Windows, MacOS, Linux or even a mobile application) for your AI applications.

## Installation of Electron

```bash
npm i -D electron@latest
```

## Creating Electron Project

### Critical Files

You should have 3 base files `package.json`, `main.js` and `index.html` to have a basic application.

````bash
mkdir app
cd app

npm init

touch main.js
touch index.html

````

### Edit package.json
- When you run `npm init`, it should create a `package.json` file. But we need to make some tiny changes to leverage on electron.
- Key fields
    - `name`: name of your app, can be anything
    - `version`: version of your app, can be anything
    - `main`: main javascript file, we recommend using `main.js`
    - `scripts`: here you want to copy the whole `scripts` section to leverage on electron
    - `devDependencies`: electron version required

```json
{
  "name": "dlw",
  "version": "0.1.0",
  "main": "main.js",
  "scripts": {
    "start": "electron ."
  },
  "devDependencies": {
    "electron": "^6.0.8"
  }
}
``` 

### Edit main.js
This beautiful boilerplate code is provided by Electron, full credits to them. 

```javascript
const { app, BrowserWindow } = require('electron')

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let win

function createWindow () {
  // Create the browser window.
  win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true
    }
  })

  // and load the index.html of the app.
  win.loadFile('index.html')

  // Open the DevTools.
  // win.webContents.openDevTools()

  // Emitted when the window is closed.
  win.on('closed', () => {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    win = null
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', () => {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  // On macOS it's common to re-create a window in the app when the
  // dock icon is clicked and there are no other windows open.
  if (win === null) {
    createWindow()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
```

### Edit index.html

I modified this script from electron's boilerplate code where it will display critical dependencies' versions for your node, chrome and electron.


```html
<!DOCTYPE html>
<html>

  <head>
    <meta charset="UTF-8">
    <title>Dashboard</title>
  </head>

  <body>
    <h1>Dashboard</h1>

    <h2>Environment</h2>
        <br/>
        Node: <script>document.write(process.versions.node)</script>

        <br/>
        Chrome: <script>document.write(process.versions.chrome)</script>

        <br/>
        Electron: <script>document.write(process.versions.electron)</script>
  </body>
</html>

```

## Starting App
This will start your electron application.

```bash
npm start
```

## Packaging Electron App

### Wine
The reason for installing Wine is being able to package Electron applications for the Windows platform, creating the executable file `app.exe` like any other application on Windows. The final aim of our tutorial is to package the app for Windows, MacOS and Ubuntu.

#### Installation of Wine
This assumes installation on Ubuntu 16.04 `xenial`, if you're on Ubuntu 18.04 or 19.04, change to `bionic` and `disco` respectively.

Also, this works on 64-bit system architecture.

```bash
cd ~
wget -qO - https://dl.winehq.org/wine-builds/winehq.key | sudo apt-key add -
sudo apt-add-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ xenial main'
sudo apt-get update
sudo apt-get install --install-recommends winehq-stable

sudo chown root:root ~/.wine
```

#### Check Wine Version
```bash
wine --version
```

### Packaging Windows Application
This packages the application churning the necessary files and the executable `app.exe` for windows 64 bit.

```bash
electron-packager ./app app --platform=win32 --arch=x64
```

## Python Scripts

### Installing Python Node Package

So we want to easily create Python scripts and run through Javascript in the Electron application. This can be done via `python-shell` npm package.

```bash
sudo npm install --save python-shell 
```

### Creating "Hello from JS"

#### Javascript

In your `main.js` file, you would want to add the following code. 

This leverages on the `python-shell` package to send a message to `hello_world.py` and receive the message subsequently.

```javascript

// Start Python shell
let {PythonShell} = require('python-shell')

// Start shell for specific script for communicating
let pyshell = new PythonShell('./scripts/hello_world.py');

// Send a message to the Python script via stdin
pyshell.send('Hello from JS');

// Receive message from Python script
pyshell.on('message', function (message) {
  console.log(message);

});

// End the input stream and allow the process to exit
pyshell.end(function (err, code, signal) {
  if (err) throw err;
//  console.log('The exit code was: ' + code);
//  console.log('The exit signal was: ' + signal);
  console.log('finished');
});

```

#### Python

Create a folder `scripts` to hold all your Python scripts. Then create a Python file named `hello_world.py` with the following content.

```python
import sys

msg_from_js = sys.stdin.read()

print(msg_from_js)
```

#### Run App

Run via `npm start` and you'll see this in your bash output. Viola! We managed to call `hello_world.py` via `main.js` through the `python-shell` package. Next task, we will be passing this message to `index.html`.

```bash
Hello from JS

finished
```