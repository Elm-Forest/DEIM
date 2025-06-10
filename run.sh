CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7777 --nproc_per_node=2 train.py -c configs/deim_dfine/deim_hgnetv2_n_coco.yml --use-amp --seed=0

pip install -r requirements.txt

CUDA_VISIBLE_DEVICES=0 python train.py -c configs/deim_dfine/deim_hgnetv2_n_coco.yml -d "cuda:0" --use-amp --seed=0