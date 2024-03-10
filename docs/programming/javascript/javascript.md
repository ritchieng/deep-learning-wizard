---
comments: true
---

# Javascript

!!! tip "Run Jupyter Notebook"
    You can run the code for this section in this [jupyter notebook link](https://github.com/ritchieng/deep-learning-wizard/blob/master/docs/programming/javascript/javascript.ipynb).

## Installation of iJavascript

This will enable the Javascript kernel to be installed in your jupyter notebook kernel list so you can play with Javascript easily.

For now, the installation requires the ancient Python 2.7. Hopefully this changes in the future.

```
conda create -n py27 python=2.7
conda activate py27
sudo npm install -g --unsafe-perm ijavascript
conda install jupyter
ijsinstall
jupyter notebook
```

## Variables & Constants

### Variable

#### Declaring Variable


```javascript
var randomNumber = 1.52
```

#### Read out Variable


```javascript
// Check value
randomNumber
```




    1.52



#### Read out via Console


```javascript
// Using the console via log and error
console.log(randomNumber)
console.error(randomNumber)
```

    1.52


    1.52


### Constants

### Declaring Constant
This is a weird one, because in Javascript when you declare a constant, you are essentially unable to overwrite the constant subsequently. It's useful when you want a fixed value.


```javascript
const cannotOverwriteNumber = 2.22
```

#### Read out Constant


```javascript
cannotOverwriteNumber
```




    2.22



#### Overwrite Constant (Error)
This will throw an error because this constant has been declared once and you cannot do it again.


```javascript
const cannotOverwriteNumber = 1.52
```


    evalmachine.<anonymous>:1

    const cannotOverwriteNumber = 1.52

    ^

    

    SyntaxError: Identifier 'cannotOverwriteNumber' has already been declared

        at evalmachine.<anonymous>:1:1

        at Script.runInThisContext (vm.js:122:20)

        at Object.runInThisContext (vm.js:329:38)

        at run ([eval]:1054:15)

        at onRunRequest ([eval]:888:18)

        at onMessage ([eval]:848:13)

        at process.emit (events.js:198:13)

        at emit (internal/child_process.js:832:12)

        at process._tickCallback (internal/process/next_tick.js:63:19)



```javascript

```
