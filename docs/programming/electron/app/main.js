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
  win.webContents.openDevTools()

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

//let {PythonShell} = require('python-shell')
//
//
//// Hello world python
//let options = {
//  mode: 'text',
//  pythonOptions: ['-u'], // get print results in real-time
//  scriptPath: 'scripts/',
//  args: ['value1', 'value2', 'value3']
//};
//
//PythonShell.run(
//  'hello_world.py',
//  options,
//  function (err, results) {
//    if (err) throw err;
//    console.log(results);
//  }
//);


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
