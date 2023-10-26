#!/bin/bash

# define arguments with getopts command
while getopts ":m:" opt; do
  case $opt in
    m)
      mode="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# default values for mode and filename
default_mode="training"

# set default values if mode and filename are not provided
mode="${mode:-$default_mode}"

# run script
python ../main.py --mode $mode &> ../MAKE/"$mode".out
