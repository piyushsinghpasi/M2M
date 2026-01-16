
cdists=(30 35 40 44 48)

ckls=(0.5 1 2)
for cdist in "${cdists[@]}"; do
	for ckl in "${ckls[@]}"; do

		WANDB_MODE=online CUDA_VISIBLE_DEVICES=1 python main.py --wandb_name 250K_mse"$cdist"_klclip"$ckl"_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse_klclip --mode train --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/"  --c_dist "$cdist" --c_kl_clip "$ckl"


	done
done
