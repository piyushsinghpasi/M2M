#mode="eval_M30K_gaussian"
mode="train"

#wandbMode="disabled"
wandbMode="online"
topk=5

# mse + L_str
cdists=(48 1 5)

ckls=(1 )
for cdist in "${cdists[@]}"; do
	for ckl in "${ckls[@]}"; do

		WANDB_MODE=$wandbMode CUDA_VISIBLE_DEVICES=1 python main.py --wandb_name 250K_mse"$cdist"_topK"$topk"_klclip"$ckl"_warmup1000_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method topk --topk $topk --loss_method mse_klclip_warmup --mode $mode --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/"  --c_dist "$cdist" --c_kl_clip "$ckl"

		# Wait for user input
		#read -p "Press Enter to continue..."

		# Next command
		echo "Continuing with the next step..."

	done
done
