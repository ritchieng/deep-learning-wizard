#!/usr/bin/env bash

day=$(date +%u)

if [ $day == 5 ];
    then
        echo "Friday is here!"

    else
        echo "Friday is not here :("
        echo "Today is day $day of the week"
    fi