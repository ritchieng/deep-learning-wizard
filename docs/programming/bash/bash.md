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

# Multiple Operations: Divide and Add
echo $((10/10 + 10))
```

```bash
20
0
1
100
0
11
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

## For Loop With Range

This will echo the digits 0 to 10 without explicitly requiring to define the whole range of numbers/alphabets like above.

```bash
for ((i=0; i<=10; i++));
    do
        echo $i
    done
```

```bash
0
1
2
3
4
5
6
7
8
9
10
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

### Sequentially Running of Python Scripts

This snippet allows you to to run 3 python scripts sequentially, waiting for each to finish before proceeding to the next.

```bash
python script_1.py
wait

python script_2.py 
wait

python script_3.py
wait

echo "Finished running all 3 scripts in sequence!"
``` 

### Parallel Running of Python Scripts

```bash
python script_1.py && script_2.py && script_3.py
wait

echo "Finished running all 3 scripts in parallel in sequence"
```

## Reading and Writing Operations


### Reading logs and texts
Create a text file called `random_text.txt` with the following contents

```text
Row 1
Row 2
Row 3
Row 4
Row 5
Row 6
Row 7
Row 8
Row 9
Row 10
```

Then run the following command to read it in bash then print it.

```bash
text_file=$(cat random_text.txt)
echo $text_file
```

```text
Row 1 Row 2 Row 3 Row 4 Row 5 Row 6 Row 7 Row 8 Row 9 Row 10
```


## Date Operations

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

### Changing System Dates By 1 Day
You can change system dates based on this. Surprisingly, you'll find it useful for testing an environment for deployments in the next day and then shifting it back to the actual day.

```bash
date -s 'next day'
date -s 'yesterday'
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

## Conda Commands

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

### Recovering problematic conda installation
```bash

# Download miniconda according to your environment
# Link: https://docs.conda.io/en/latest/miniconda.html

# Backup existing miniconda environment that may have problems
mv miniconda3 miniconda3_backup

# Install miniconda
bash Miniconda3-latest-Linux-x86_64.sh

# Restore old environment settings
rsync -a miniconda3_backup/ miniconda3/

```

## Internet Operations

### Checking Internet Availability

This script will return whether your internet is fine or not without using pings.
 
Pings can often be rendered unusable when the network administrator disables ICMP to prevent the origination of ping floods from the data centre.

```bash
if nc -zw1 google.com 443;
    then
        echo "INTERNET: OK"
    else
        echo "INTERNET: NOT OK"
    fi
```