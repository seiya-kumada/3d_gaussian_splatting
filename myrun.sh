#!/bin/bash
# cmake .. -DCMAKE_PREFIX_PATH=~/projects/libtorch
./build/train \
	--source_path hoge \
	--white_background \
	--eval \
	--model_path /home/ubuntu/data/3d_gaussian_splatting/model \
