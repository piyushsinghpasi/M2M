
# multi30K with pretrained multilingual models
#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode "Other_M-CLIP/LABSE-Vit-L-14-multi30k" --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_jina_jinaTextv3/config.json

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode "Other_M-CLIP/XLM-Roberta-Large-Vit-B-32-multi30k" --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_jina_jinaTextv3/config.json

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode "Other_M-CLIP/XLM-Roberta-Large-Vit-L-14-multi30k" --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_jina_jinaTextv3/config.json

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode "Other_M-CLIP/XLM-Roberta-Large-Vit-B-16Plus-multi30k" --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_jina_jinaTextv3/config.json

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode "Other_M-jina-multi30k" --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_jina_jinaTextv3/config.json

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode "Other_M-sentCLIP-multi30k" --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_jina_jinaTextv3/config.json


# baseline monolingual model Multi30K 
#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode M30K-baseline-clip --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_cliplarge_multiMpnet/config.json 

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode M30K-baseline-jina --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_jina_multiMpnet/config.json

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode M30K-baseline-align --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_align_multiMpnet/config.json 



# our models with Multi30K
## Text ablations

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode M30K-jina-Labse --config ./content/drive/MyDrive/base-clip-data/models/250K_mse_klclip_jina/config.json

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode M30K-jina-miniLM --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_jina_multiMiniLM/config.json

CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode M30K-jina-jinav3 --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_jina_jinaTextv3/config.json

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode M30K-jina-mpnet --config ./content/drive/MyDrive/base-clip-data/models/250K_mse48_klclip1_jina_multiMpnet/config.json


## image ablations
#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode M30K-cliplarge-mpnet --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_cliplarge_multiMpnet/config.json

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode M30K-align-mpnet --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_align_multiMpnet/config.json


# Crossmodal 3600
# monolingual baselines
#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode crossmodal3600-baseline-clip --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_cliplarge_multiMpnet/config.json 

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode crossmodal3600-baseline-jina --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_jina_multiMpnet/config.json

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode crossmodal3600-baseline-align --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_align_multiMpnet/config.json 

## Our text models with crossmodal3600

#WANDB_MODE=disabled python main.py --mode crossmodal3600-jina-Labse --config ./content/drive/MyDrive/base-clip-data/models/250K_mse_klclip_jina/config.json

#WANDB_MODE=disabled python main.py --mode crossmodal3600-jina-miniLM --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_jina_multiMiniLM/config.json

WANDB_MODE=disabled python main.py --mode crossmodal3600-jina-jinav3 --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_jina_jinaTextv3/config.json

#WANDB_MODE=disabled python main.py --mode crossmodal3600-jina-mpnet --config ./content/drive/MyDrive/base-clip-data/models/250K_mse48_klclip1_jina_multiMpnet/config.json


## image ablations
#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode crossmodal3600-cliplarge-mpnet --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_cliplarge_multiMpnet/config.json

#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode crossmodal3600-align-mpnet --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_align_multiMpnet/config.json

