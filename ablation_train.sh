
ckl=1
cdist=48
device=0

wandb_mode="disabled"
#wandb_mode="online"

mode="eval_labse"
#mode="train"

# labse
WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name 250K_mse_klclip_jina --image_model jinav1 --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse_klclip --mode "$mode" --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/" --c_dist "$cdist" --c_kl_clip "$ckl"

WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name en_250K_mse_klclip_jina_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model multiMpnet --mode "$mode" --train_langs en_250K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre" --c_dist "$cdist" --c_kl_clip "$ckl"

WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name en_250K_mse_klclip_jina_jinaTextv3 --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model jinaTextv3 --mode "$mode" --train_langs en_250K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre" --c_dist "$cdist" --c_kl_clip "$ckl"

source /home/darshan/internet_access.sh

WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name en_250K_mse_klclip_jina_multiMiniLM --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model jinav1 --text_model multiMiniLM --mode "$mode" --train_langs en_250K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre" --c_dist "$cdist" --c_kl_clip "$ckl"

WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name en_250K_mse_klclip_cliplarge_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model clip --text_model multiMpnet --mode "$mode" --train_langs en_250K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre" --c_dist "$cdist" --c_kl_clip "$ckl"

WANDB_MODE="$wandb_mode" CUDA_VISIBLE_DEVICES="$device" python main.py --wandb_name en_250K_mse_klclip_align_multiMpnet --epochs 50 --num_batch 100000000000 --emb_method no_skip_conn --loss_method mse_klclip --num_layers 2 --image_model align --text_model multiMpnet --mode "$mode" --train_langs en_250K --eng_base_path "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre" --c_dist "$cdist" --c_kl_clip "$ckl"

