ckl=1
cdist=48
device=1
wandb_mode=disabled
mode="eval_labse"

WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name 250K_mse_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse --mode "$mode" --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/" --c_dist "$cdist" --c_kl_clip "$ckl"

WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name 250K_mse_skip_conn_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method skip_conn --loss_method mse --mode "$mode" --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/" --c_dist "$cdist" --c_kl_clip "$ckl"

WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name 250K_mse_klclip_skip_conn_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method skip_conn --loss_method mse_klclip --mode "$mode" --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/" --c_dist "$cdist" --c_kl_clip "$ckl"

WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name 250K_mse_klclip_L4_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --num_layers 4 --loss_method mse_klclip --mode "$mode" --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/" --c_dist "$cdist" --c_kl_clip "$ckl"

source /home/darshan/internet_access.sh

WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name en_250_mse_klclip_L1_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 1 --image_model jinav1 --text_model multiMpnet --mode "$mode" --train_langs en_250K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre" --c_dist "$cdist" --c_kl_clip "$ckl"

WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name en_250_eng_cosine_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method eng_cosine --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode "$mode" --train_langs en_250K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre" --c_dist "$cdist" --c_kl_clip "$ckl"

WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name en_250_l1_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method l1 --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode "$mode" --train_langs en_250K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre" --c_dist "$cdist" --c_kl_clip "$ckl"

