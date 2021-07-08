script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)
ip_list=$(cat $script_dir/ip_list.txt)
docker run --gpus all -d --ipc=host --cap-add=IPC_LOCK -v /sys/class/net/:/sys/class/net/  --device=/dev/ --privileged --network=host -v $main_dir:/root/cogview2 -v /mnt/:/root/mnt baai:deepspeed-jit bash -c "/etc/init.d/ssh start && python /root/cogview2/docker/setup_connection.py $ip_list && sleep 365d"
