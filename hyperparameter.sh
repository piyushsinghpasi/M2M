
cdists=(1 5 10 20 30 40 48 60 80)
cdists=(48)

ckls=(1 )
for cdist in "${cdists[@]}"; do
	for ckl in "${ckls[@]}"; do

		WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name 250K_mse"$cdist"_klclip"$ckl"_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method no_skip_conn --loss_method mse_klclip --mode eval_labse --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/"  --c_dist "$cdist" --c_kl_clip "$ckl"

		# Wait for user input
		read -p "Press Enter to continue..."

		# Next command
		echo "Continuing with the next step..."

	done
done
