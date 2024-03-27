NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4  --master_port 29502 train.py -p 29502 -d 0,1,2,3

# CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7" torchrun -m --nproc_per_node=1 train.py -p 29501 -d 7