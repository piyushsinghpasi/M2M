
# baseline monolingual model Multi30K 


#WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python3 main.py --wandb_name 250K_mse48_klclip1_siglip1_multiMpnet --image_model siglip1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse_klclip --mode M30K-baseline-siglip1 --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/" --c_dist 48 --c_kl_clip 1

# Our model M30K

#WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python3 main.py --wandb_name 250K_mse48_klclip1_siglip1_multiMpnet --image_model siglip1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse_klclip --mode M30K-siglip1 --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/" --c_dist 48 --c_kl_clip 1



# Crossmodal 3600
# monolingual baselines


#WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python3 main.py --wandb_name 250K_mse48_klclip1_siglip1_multiMpnet --image_model siglip1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse_klclip --mode crossmodal3600-baseline-siglip1 --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/" --c_dist 48 --c_kl_clip 1

## Our text models with crossmodal3600



WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=1 python3 main.py --wandb_name 250K_mse48_klclip1_siglip1_multiMpnet --image_model siglip1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse_klclip --mode crossmodal3600-siglip1-mpnet --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/" --c_dist 48 --c_kl_clip 1

## image ablations
#CUDA_VISIBLE_DEVICES=1 WANDB_MODE=disabled python main.py --mode crossmodal3600-align-mpnet --config ./content/drive/MyDrive/base-clip-data/models/en_250K_mse_klclip_align_multiMpnet/config.json

