
cdist=44
ckl=1

#WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name 250K_mse"$cdist"_klclip"$ckl"_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse_klclip --mode eval_labse --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/"  --c_dist "$cdist" --c_kl_clip "$ckl"


WANDB_MODE=online CUDA_VISIBLE_DEVICES=0 python main_irish.py \
	--wandb_name 250K_mse_irish_jina_labse \
	--image_model jinav1 \
	--epochs 50 \
	--num_batch 1000000000 \
	--emb_method no_skip_conn \
	--loss_method mse_non_eng \
	--mode train \
	--train_langs xxx \
	--eng_base_path yyy \
       	--base_image_dir "/home/piyush/labse-clip/irish-data/wit_ga_images/" 

