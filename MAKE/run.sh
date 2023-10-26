#!/bin/bash

# Define default values
mode="training"
batch_size=256
lr=0.001
weight_decay=0.0001
device="cuda"
num_epochs=2

# Parse command-line options
OPTIONS=$(getopt -o m:b:l:w:d:e: --long mode:,batch_size:,lr:,weight_decay:,device:,num_epochs: -n 'script.sh' -- "$@")
if [ $? -ne 0 ]; then
    echo "Usage: $0 -m <mode> -b <batch_size> -l <lr> -w <weight_decay> -d <device> -e <num_epochs>"
    exit 1
fi

eval set -- "$OPTIONS"

while true; do
    case "$1" in
        -m|--mode)
            mode="$2"
            shift 2
            ;;
        -b|--batch_size)
            batch_size="$2"
            shift 2
            ;;
        -l|--lr)
            lr="$2"
            shift 2
            ;;
        -w|--weight_decay)
            weight_decay="$2"
            shift 2
            ;;
        -d|--device)
            device="$2"
            shift 2
            ;;
        -e|--num_epochs)
            num_epochs="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

# Run script with the provided arguments
python ../main.py --mode "$mode" --batch_size "$batch_size" --lr "$lr" --weight_decay "$weight_decay" --device "$device" --num_epochs "$num_epochs" &> "../MAKE/$mode.out"
