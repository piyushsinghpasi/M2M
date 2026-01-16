# NOTE: cdist = 48 and ckl = 1 by default in main

echo HTSAT CLAP with Audiocaps test
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python main_clap.py --wandb_name clap_ht_fused_multiMpnet_audiocaps_clotho_wavcaps --mode eval_labse --num_batch 1000000000000 --epochs 50 --emb_method "no_skip_conn" --loss_method mse_klclip --train_langs audiocaps_train_45K clotho_train_from_wavcaps_19K wavcaps_train_304K --audio_model clap --text_model multiMpnet --eng_base_path ./content/drive/MyDrive/clip-data/AWS\ 68\ Languages/unique_texts_pre/ --img_to_text_map_path ./eval_data/audio_dataset/audiocaps/dataset/audiocaps_test_33_language_translation_cleaned_img_to_text_map.pkl --text_to_img_map_path ./eval_data/audio_dataset/audiocaps/dataset/audiocaps_test_33_language_translation_cleaned_text_to_img_map.pkl --audio_base_dir ./eval_data/audio_dataset/test/ --eval_data_path test_audiocaps


echo General CLAP with Audiocaps test
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python main_clap.py --wandb_name clap_general_multiMpnet_audiocaps_clotho_wavcaps --mode eval_labse --num_batch 1000000000000 --epochs 50 --emb_method "no_skip_conn" --loss_method mse_klclip --train_langs audiocaps_train_45K clotho_train_from_wavcaps_19K wavcaps_train_304K --audio_model clap_general --text_model multiMpnet --eng_base_path ./content/drive/MyDrive/clip-data/AWS\ 68\ Languages/unique_texts_pre/ --img_to_text_map_path ./eval_data/audio_dataset/audiocaps/dataset/audiocaps_test_33_language_translation_cleaned_img_to_text_map.pkl --text_to_img_map_path ./eval_data/audio_dataset/audiocaps/dataset/audiocaps_test_33_language_translation_cleaned_text_to_img_map.pkl --audio_base_dir ./eval_data/audio_dataset/test/ --eval_data_path test_audiocaps



echo HTSAT CLAP with CLOTHO test
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python main_clap.py --wandb_name clap_ht_fused_multiMpnet_audiocaps_clotho_wavcaps --mode eval_labse --num_batch 1000000000000 --epochs 50 --emb_method "no_skip_conn" --loss_method mse_klclip --train_langs audiocaps_train_45K clotho_train_from_wavcaps_19K wavcaps_train_304K --audio_model clap --text_model multiMpnet --eng_base_path ./content/drive/MyDrive/clip-data/AWS\ 68\ Languages/unique_texts_pre/ --img_to_text_map_path ./eval_data/audio_dataset/CLOTHO/clotho_33_languages_translations_cleaned_img_to_text_map.pkl --text_to_img_map_path ./eval_data/audio_dataset/CLOTHO/clotho_33_languages_translations_cleaned_text_to_img_map.pkl --audio_base_dir ./eval_data/audio_dataset/CLOTHO/evaluation/ --eval_data_path test_clotho


echo General CLAP with CLOTHO test
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python main_clap.py --wandb_name clap_general_multiMpnet_audiocaps_clotho_wavcaps --mode eval_labse --num_batch 1000000000000 --epochs 50 --emb_method "no_skip_conn" --loss_method mse_klclip --train_langs audiocaps_train_45K clotho_train_from_wavcaps_19K wavcaps_train_304K --audio_model clap_general --text_model multiMpnet --eng_base_path ./content/drive/MyDrive/clip-data/AWS\ 68\ Languages/unique_texts_pre/ --img_to_text_map_path ./eval_data/audio_dataset/CLOTHO/clotho_33_languages_translations_cleaned_img_to_text_map.pkl --text_to_img_map_path ./eval_data/audio_dataset/CLOTHO/clotho_33_languages_translations_cleaned_text_to_img_map.pkl --audio_base_dir ./eval_data/audio_dataset/CLOTHO/evaluation/ --eval_data_path test_clotho


