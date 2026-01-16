
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel
from sentence_transformers import SentenceTransformer

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import torch
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
import argparse

import wandb
import random
import json
import pickle


from clip_base.model import labse_clip
from clip_base.dataloader import TrainDataset, EmbeddingPrecomputator
from trainer import Trainer

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LABSE_MODEL_NAME = "sentence-transformers/LaBSE"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Labse-CLIP")
    parser.add_argument("--wandb_name", type=str, help="wandb run name")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_batch", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_min", type=float, default=1e-8)
    parser.add_argument("--emb_method", type=str, default="skip_conn", 
            help="used in model for skip conn or sequential transformation")
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--loss_method", type=str, default="mse", help="support diff losses see trainer.py")
    parser.add_argument("--mode", type=str, default="eval_clip", help="train, baseline_clip, baseline_labse, eval_labse (Trained checkpoint)")
    parser.add_argument("--train_langs", type=str, nargs="+", 
            default=['hi', 'ta', 'bn', 'te', 'ru', 'fr'], 
            help="langs used in KL for training, and their eq. english is used in mse")
    parser.add_argument("--c_dist", type=int, default=44, 
            help="support diff losses see trainer.py, "
            "used for linear comb. wt. for mse/l1")
    parser.add_argument("--c_kl", type=int, default=1, 
            help="support diff losses see trainer.py, "
            "used for linear comb. wt. for kl")
    parser.add_argument("--c_kl_clip", type=int, default=1, 
            help="support diff losses see trainer.py, "
            "used for linear comb. wt. for kl clip")
    parser.add_argument("--early_stopping", action="store_true", help="earlystopping")
    parser.add_argument("--penalty_gap", type=int, default=10, 
            help="epochs to wait after best epoch before triggering early stopping")
    parser.add_argument("--save_dir", type=str, help="dir where checkpoint", 
            default="./content/drive/MyDrive/base-clip-data/models")

    args= parser.parse_args()

    wandb.init(project="Labse-CLIP", name=args.wandb_name, config=args)

    os.makedirs(args.save_dir, exist_ok=True)


    if args.emb_method == "hybrid":
        from model_hybrid import labse_clip

    """# RUN

    ## Load model
    """

    device = "cuda"
    model = labse_clip(hdim=768, args=args)
    model = model.to(device)

    """## Load training data"""

    eng_base_path = "./content/drive/MyDrive/clip-data/AWS 68 Languages/Pre/"
    non_eng_base_path = "./content/drive/MyDrive/clip-data/AWS 68 Languages/Post/"
    #langs = ['hi', 'ta', 'bn', 'te', 'ru', 'fr']
    langs = args.train_langs
    eng_file_paths = [os.path.join(eng_base_path, lang + '.txt') for lang in langs]
    non_eng_file_paths = [os.path.join(non_eng_base_path, lang + '.txt') for lang in langs]

    train_data = TrainDataset(
        eng_file_paths=eng_file_paths,
        non_eng_file_paths=non_eng_file_paths,
        labse_model=model.labse,
        clip_model=model.clip_model,
        clip_processor=model.clip_processor,
        clip_text_to_id_path = "./content/drive/MyDrive/base-clip-data/clip_text_to_id_train_base_clip.pkl" ,
        labse_text_to_id_path = "./content/drive/MyDrive/base-clip-data/labse_text_to_id_train.pkl",
        feature_save_dir = "./content/drive/MyDrive/base-clip-data/features",
    )

    # implies 100 batches per epoch
    K = args.batch_size*args.num_batch

    if K < len(train_data.eng_data):
        print(f"Sampling {K} instances to train")
        
        random.seed(42)
        indices = random.sample(range(len(train_data.eng_data)), K)

        train_data.eng_data = [train_data.eng_data[i] for i in indices]
        train_data.non_eng_data = [train_data.non_eng_data[i] for i in indices]
    else:
        print("Training on all data...")

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    """## Load Test data"""


    eval_data_path = "./content/drive/MyDrive/base-clip-data/Cross-lingual-Test-Dataset-XTD10/merged_data.csv"
    img_text_df = pd.read_csv(eval_data_path)

    # below path is used to load and save text mappings
    # only create for first run then load by setting load_features=True
    text_to_id_path = "./content/drive/MyDrive/base-clip-data/text_to_id.pkl"
    os.makedirs(os.path.dirname(text_to_id_path), exist_ok=True)

    eval_dataset = EmbeddingPrecomputator(
        img_text_df,
        model.clip_model,
        model.clip_processor,
        model.labse,
        device,
        image_dir = "./content/drive/MyDrive/base-clip-data/images",
        feature_save_dir = "./content/drive/MyDrive/base-clip-data/features",
        # lang_cols = ['XTD10_captions_en'],
        lang_cols = ['XTD10_captions_it',
           'XTD10_captions_ru', 'XTD10_captions_en', 'XTD10_captions_pl',
           'XTD10_captions_tr', 'XTD10_captions_zh', 'XTD10_captions_ko',
           'XTD10_captions_es', 'STAIR_caption_jp', 'MIC_caption_de',
           'MIC_caption_fr'],
        img_col = 'test_image_names',
        text_to_id_path = text_to_id_path,
    )

    """# Training"""

    num_train_steps = len(train_loader) * args.epochs  # Adjust according to your dataloader

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)


    # Create scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_train_steps,
    )

    trainer = Trainer(
        model,
        train_loader,
        eval_dataset,
        optimizer,
        scheduler,
        device,
        [1, 5, 10],
        args,
    )

    if args.mode == "train":
        trainer.train(1, args.epochs)
    elif args.mode == "baseline_labse":
        eval_dataset = EmbeddingPrecomputator(
            img_text_df,
            model.clip_model,
            model.clip_processor,
            model.labse,
            device,
            image_dir = "./content/drive/MyDrive/base-clip-data/images",
            feature_save_dir = "./content/drive/MyDrive/base-clip-data/features",
            # lang_cols = ['XTD10_captions_en'],
            lang_cols = ['XTD10_captions_it',
               'XTD10_captions_ru', 'XTD10_captions_en', 'XTD10_captions_pl',
               'XTD10_captions_tr', 'XTD10_captions_zh', 'XTD10_captions_ko',
               'XTD10_captions_es', 'STAIR_caption_jp', 'MIC_caption_de',
               'MIC_caption_fr'],
            img_col = 'test_image_names',
            text_to_id_path = text_to_id_path,
        )

        trainer.eval_dataset = eval_dataset
        metrics = trainer.evaluate(labse_baseline=True)
        with open("./metrics/baseline_labse_clip_base.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print("Metrics saved at: ./metrics/baseline_labse_clip_base.json")
    elif args.mode == "baseline_clip":
        eval_dataset = EmbeddingPrecomputator(
            img_text_df,
            model.clip_model,
            model.clip_processor,
            model.labse,
            device,
            image_dir = "./content/drive/MyDrive/base-clip-data/images",
            feature_save_dir = "./content/drive/MyDrive/base-clip-data/features",
            lang_cols = ['XTD10_captions_en'],
            img_col = 'test_image_names',
            text_to_id_path = text_to_id_path,
        )
        trainer.eval_dataset = eval_dataset
        metrics = trainer.evaluate(use_clip=True)
        with open("./metrics/baseline_clip_base.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print("Metrics saved at: ./metrics/baseline_clip_base.json")
    elif args.mode == "eval_labse":
        save_path = os.path.join(args.save_dir, args.wandb_name+".pth")
        loaded_layers = torch.load(save_path, map_location='cpu')
        
        for i, layer in enumerate(trainer.model.mlps):
            layer.load_state_dict(
                loaded_layers[f'layer_{i}']
            )
         
        eval_dataset = EmbeddingPrecomputator(
            img_text_df,
            model.clip_model,
            model.clip_processor,
            model.labse,
            device,
            image_dir = "./content/drive/MyDrive/base-clip-data/images",
            feature_save_dir = "./content/drive/MyDrive/base-clip-data/features",
            lang_cols = ['XTD10_captions_it',
               'XTD10_captions_ru', 'XTD10_captions_en', 'XTD10_captions_pl',
               'XTD10_captions_tr', 'XTD10_captions_zh', 'XTD10_captions_ko',
               'XTD10_captions_es', 'STAIR_caption_jp', 'MIC_caption_de',
               'MIC_caption_fr'],
            img_col = 'test_image_names',
            text_to_id_path = text_to_id_path,
        )


        trainer.eval_dataset = eval_dataset
        metrics = trainer.evaluate()
        with open(f"./metrics/{args.wandb_name}_labse_clip_base.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved at: ./metrics/{args.wandb_name}_labse_clip_base.json")
