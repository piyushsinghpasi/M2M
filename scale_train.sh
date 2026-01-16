

# scale exps
WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_1K_mse_klclip_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode train --train_langs en_1K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_5K_mse_klclip_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode train --train_langs en_5K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_10K_mse_klclip_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode train --train_langs en_10K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_50K_mse_klclip_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode train --train_langs en_50K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_100K_mse_klclip_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode train --train_langs en_100K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_250K_mse_klclip_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode train --train_langs en_250K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"

WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name en_2M_mse_klclip_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode train --train_langs en_2M --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"





