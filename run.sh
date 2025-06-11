CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=7777 --nproc_per_node=2 train.py -c configs/deim_dfine/deim_hgnetv2_n_coco.yml --use-amp --seed=0

pip install -r requirements.txt
pip install -r tools/inference/requirements.txt

CUDA_VISIBLE_DEVICES=0 python train.py -c configs/deim_dfine/deim_hgnetv2_n_coco.yml -d "cuda:0" --use-amp --seed=0

python tools/inference/torch_inf.py -c configs/deim_dfine/deim_hgnetv2_s_coco.yml -r "/kaggle/working/DEIM/outputs/deim_hgnetv2_s_coco/checkpoint0115.pth" --input "/kaggle/input/cplid-insulator/Defective_Insulators/images/006.jpg" --device "cuda:0"