
# baseline siglip1
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python3 main.py --wandb_name 250K_mse48_klclip1_siglip1_multiMpnet --image_model siglip1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse_klclip --mode baseline_clip --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/" --c_dist 48 --c_kl_clip 1

# siglip1 + ours
#WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python3 main.py --wandb_name 250K_mse48_klclip1_siglip1_multiMpnet --image_model siglip1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse_klclip --mode eval_labse --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/" --c_dist 48 --c_kl_clip 1


#WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name 250K_mse48_klclip1_siglip1_multiMpnet --image_model siglip2_so400 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse_klclip --mode eval_labse --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/" --c_dist 48 --c_kl_clip 1
