# Bash

!!! tip "Run Bash Files"
    You can find the bash code files for this section in this [link](https://github.com/ritchieng/deep-learning-wizard/blob/master/docs/programming/bash).

## Creating and Running a Bash File

### Create bash file
```bash
touch hello_world.sh
```

### Edit bash file with Hello World 
You can edit with anything like Vim, Sublime, etc.

```bash
echo Hello World!
```

### Run bash file
```bash
bash hello_world.sh
```

This will print out, in your bash:
```bash
Hello World!
```

## Calculating

```bash
# Add
echo $((10+10))

# Subtract
echo $((10-10))

# Divide
echo $((10/10))

# Multiple
echo $((10*10))

# Modulo
echo $((10%10))
```

```bash
20
0
1
100
0
```

## Getting Dates


### Getting Current Date
This will return the date in the format YYYY-MM-DD for example 2019-06-03.

```bash
DATE=`date +%Y-%m-%d`
echo $DATE
```

### Getting Current Day of Week
This will return 1, 2, 3, 4, 5, 6, 7 depending on the day of the week.

```bash
DAY=$(date +%u)
echo $DAY
```

## Loops and Conditional

### For Loop

```bash
for i in 'A' 'B' 'C'
    do
        echo $i
    done
```

```bash
A
B
C
```

### If Else Conditional

This is a simple if-else to check if the day of the week is 5, meaning if it is a Friday.

```bash
day=$(date +%u)

if [ $day == 5 ];
    then
        echo "Friday is here!"

    else
        echo "Friday is not here :("
        echo "Today is day $day of the week"
    fi
```


## Jupyter Utility Commands

### Convert Notebook to HTML/Markdown
```bash
jupyter nbconvert --to markdown python.ipynb
jupyter nbconvert --to html python.ipynb
``` 
## Bash Convenient Commands

### List directories only
`ls -d */`

### List non-directories only
`ls -p | grep -v '/$'`

### Check IP
`ifconfig | sed -En "s/127.0.0.1//;s/.*inet (addr:)?(([0-9]*\.){3}[0-9]*).*/\2/p"`

### Check internet speed
`ethtool eno1`

### Check disk space
`df -h`

### Check ubuntu version
`lsb_release -a`

### Check truncated system logs
`tail /var/log/syslog`

### Check CUDA version
`nvcc -V`

### Check cuDNN version
`cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`

### Check conda environment
`conda env list`

### Create conda kernel
```bash
conda create -n kernel_name python=3.6
source activate kernel_name
```

### Install conda kernel
```bash
conda install ipykernel
source activate kernel_name
python -m ipykernel install --user --name kernel_name --display-name kernel_name
```

### Remove conda kernel
`conda env remove -n kernel_name`

### Untar file
`tar -xvzf file_name`

### Open PDF file
`gvfs-open file_name`

### Download file from link rapidly with aria
`aria2c -x16 -c url_link`

### Kill all python processes
`ps ax | grep python | cut -c1-5 | xargs kill -9`

### Install .deb files
`sudo apt-get install -f file_name.deb`

### Empty thrash
```bash
sudo apt-get install trash-cli
thrash-empty
```