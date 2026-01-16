#mode="eval_M30K_gaussian"
#mode="train"
mode="eval_labse"

wandbMode="disabled"
#wandbMode="online"
topk=5

# mse + L_str
cdists=(48 1 5)

ckls=(1 )
for cdist in "${cdists[@]}"; do
	for ckl in "${ckls[@]}"; do

		WANDB_MODE=$wandbMode CUDA_VISIBLE_DEVICES=0 python main.py --wandb_name 250K_mse"$cdist"_toplastK"$topk"_klclip"$ckl"_jina_multiMpnet --image_model jinav1 --text_model multiMpnet --epochs 50 --num_batch 1000000000 --emb_method toplastk --topk $topk --loss_method mse_klclip --mode $mode --train_langs en_250K --eng_base_path "./content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre/"  --c_dist "$cdist" --c_kl_clip "$ckl"

		# Wait for user input
		#read -p "Press Enter to continue..."

		# Next command
		echo "Continuing with the next step..."

	done
done
