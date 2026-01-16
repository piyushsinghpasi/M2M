#mode="eval_M30K_gaussian"
mode="train"

#wandbMode="disabled"
wandbMode="online"

# mse only
#WANDB_MODE=$wandbMode CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name 250K_mse_jina_multiMpnet_gaussian --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method gaussian --loss_method mse --mode $mode --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/"  --c_dist 1 --c_kl_clip 1


# mse + L_str
cdists=(48 )

ckls=(2 )
for cdist in "${cdists[@]}"; do
	for ckl in "${ckls[@]}"; do

		WANDB_MODE=$wandbMode CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name 250K_mse"$cdist"_klclip"$ckl"_jina_multiMpnet_gaussian --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method gaussian --loss_method mse_klclip --mode $mode --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/"  --c_dist "$cdist" --c_kl_clip "$ckl"

		# Wait for user input
		#read -p "Press Enter to continue..."

		# Next command
		echo "Continuing with the next step..."

	done
done
