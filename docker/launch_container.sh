#!/bin/sh
# Launch a container with monodepth2 dependencies installed

cmd_line="$@"
echo "Executing in the docker container (don-t foget to use tmux)":
echo $cmd_line

docker run -ti -d --name monodepth2 --cpus 6 --shm-size 22g --gpus all -v /HDD/Documents/raw_data_downloader-KITTI-monodepth2:/dataset -v /HDD/Documents/monodepth2:/monodepth2:rw monodepth2 $cmd_line
