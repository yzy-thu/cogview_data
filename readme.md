# CogView generative cross-modality pretraining (生成式跨模态预训练)
## Environment
Suppose that the docker `baai:deepspeed-jit` exists in your servers, which we will release soon. *TODO*
### Single Node 
1. Use `git clone` to download this repo, and make sure the data (LMDB format) are moved into the `data` subfolder.   
2. Start the docker by `./docker/start_docker.sh`.
3. Get into the docker container via `docker exec -it <container_id> bash`. `pip install lmdb`
4. Get into `/root/cogview2` and run `./scripts/pretrain_single_node.sh`. You may need to change the config in the shell script.
5. Generate samples by `./scripts/text_to_image.sh`. 
### Multiple Nodes 
If you want to train the models on multiple servers inter-connected by infiniband without a shared file system (you may need `pdsh` to accelerate this process):
1. On **each** server, use `git clone` to download this repo, and make sure the data (LMDB format) are moved into the `data` subfolder. 
2. On **each** server, `echo "172.30.0.214 172.30.0.215 <other IPs>" > ./docker/ip_list.txt`, and then Start the docker by `./docker/start_docker.sh`. `pip install lmdb`
3. Get into **the docker on the first node** container via `docker exec -it <container_id> bash`.
4. Get into `/root/cogview2` and run `./scripts/pretrain_multiple_nodes.sh`. You may need to change the config (especially `OPTIONS_NCCL`) in the shell script.
5. Generate samples by `./scripts/text_to_image.sh`. 