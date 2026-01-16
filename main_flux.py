
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


#from flux.model_simple import labse_clip
from flux.model import labse_clip
from flux.dataloader import TrainDataset, EmbeddingPrecomputator
from flux.trainer import Trainer

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

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
    parser.add_argument("--c_eng_cosine", type=int, default=1, 
            help="support diff losses see trainer.py, "
            "used for linear comb. wt. for kl clip")
    parser.add_argument("--early_stopping", action="store_true", help="earlystopping")
    parser.add_argument("--save_ranks", action="store_true", help="save predicted ranks")
    parser.add_argument("--penalty_gap", type=int, default=10, 
            help="epochs to wait after best epoch before triggering early stopping")
    parser.add_argument("--save_dir", type=str, help="dir where checkpoint", 
            default="./content/drive/MyDrive/base-clip-data/models")
    parser.add_argument("--image_model", type=str, help="image model clip/jinav1", 
            default="clip")
    parser.add_argument("--text_model", type=str, help="text model labse", 
            default="labse")

    args= parser.parse_args()
    
    if "labse" in args.text_model.lower():
        LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
    else:
        LABSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    
    wandb.init(project="Labse-CLIP", name=args.wandb_name, config=args)

    os.makedirs(args.save_dir, exist_ok=True)


    if args.emb_method == "hybrid":
        from model_hybrid import labse_clip


    print(f"Using {args.image_model} and {args.text_model}")
    """# RUN

    ## Load model
    """

    device = "cuda"
    model = labse_clip(hdim=768, args=args)
    model = model.to(device).to(dtype=torch.bfloat16)

    labse = SentenceTransformer(LABSE_MODEL_NAME)
    if "labse" in LABSE_MODEL_NAME.lower():
        print("LaBSE model used, removing normalization layer")
        labse[3] = torch.nn.Identity()
    labse = labse.to(device)
    """## Load training data"""

    #eng_base_path = "./content/drive/MyDrive/clip-data/AWS 68 Languages/Pre/"
    eng_base_path = "/home/piyush/labse-clip/content/drive/MyDrive/clip-data/AWS 68 Languages/unique_texts_pre"
    non_eng_base_path = "./content/drive/MyDrive/clip-data/AWS 68 Languages/Post/"
    #langs = ['hi', 'ta', 'bn', 'te', 'ru', 'fr']
    langs = args.train_langs
    eng_file_paths = [os.path.join(eng_base_path, lang + '.txt') for lang in langs]
    #non_eng_file_paths = [os.path.join(non_eng_base_path, lang + '.txt') for lang in langs]
    non_eng_file_paths = []

    train_data = TrainDataset(
        eng_file_paths=eng_file_paths,
        non_eng_file_paths=non_eng_file_paths,
        labse_model=labse,
        clip_model=model.clip_model,
        clip_processor=model.clip_processor,
        clip_text_to_id_path = f"./content/drive/MyDrive/base-clip-data/{args.image_model}_text_to_id_train.pkl" ,
        labse_text_to_id_path = f"./content/drive/MyDrive/base-clip-data/{args.text_model}_text_to_id_train.pkl",
        feature_save_dir = "./content/drive/MyDrive/base-clip-data/features",
        image_model = args.image_model,
        text_model = args.text_model,
        device = device,
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
        None,
        optimizer,
        scheduler,
        device,
        [1, 5, 10],
        args,
    )

    if args.mode == "train":
        trainer.train(1, args.epochs)


    # DO NOT USE BELOW BASELINES
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
            image_model = args.image_model,
            text_model = args.text_model,
        )

        trainer.eval_dataset = eval_dataset
        metrics = trainer.evaluate(labse_baseline=True)
        with open(f"./metrics/baseline_{args.text_model}.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved at: ./metrics/baseline_{args.text_model}.json")
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
            image_model = args.image_model,
            text_model = args.text_model,
        )
        trainer.eval_dataset = eval_dataset
        metrics = trainer.evaluate(use_clip=True)
        with open(f"./metrics/baseline_{args.image_model}.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved at: ./metrics/baseline_{args.image_model}.json")
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
            image_model = args.image_model,
            text_model = args.text_model,
        )


        trainer.eval_dataset = eval_dataset
        metrics = trainer.evaluate(return_top_idx=args.save_ranks)
        if args.save_ranks:
            metrics, ranks = metrics
            with open(f"./ranks/{args.wandb_name}_{args.text_model}.pkl", "wb") as f:
                pickle.dump(ranks, f)
            print(f"Ranks saved at: ./ranks/{args.wandb_name}_{args.text_model}.pkl")
        
        with open(f"./metrics/{args.wandb_name}_{args.text_model}.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved at: ./metrics/{args.wandb_name}_{args.text_model}.json")
    elif args.mode == "compare_embs":

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
            image_model = args.image_model,
            text_model = args.text_model,
        )


        trainer.eval_dataset = eval_dataset
        
        img_text_df = img_text_df.head(5)
        lang = 'XTD10_captions_en'
        
        for idx, row in img_text_df.iterrows():

            query_text = row[lang]  # Assuming 'caption' is the text query column
            clip_query_embedding = eval_dataset.get_text_embedding(query_text, lang, use_clip=True)
            labse_query_embedding = eval_dataset.get_text_embedding(query_text, lang, use_clip=False).to(device)

            labse_query_embedding = model.inference(labse_query_embedding).detach().cpu()

            clip_query_embedding = torch.nn.functional.normalize(clip_query_embedding, dim=-1, p=2)
            labse_query_embedding = torch.nn.functional.normalize(labse_query_embedding, dim=-1, p=2)
            
            print("CLIP")
            print(clip_query_embedding[:, :20])
            print("LaBSE")
            print(labse_query_embedding[:, :20])
            print("abs mean")
            print((clip_query_embedding - labse_query_embedding).abs().mean())
            print("abs sum")
            print((clip_query_embedding - labse_query_embedding).abs().sum())
            print("cosine")
            print((clip_query_embedding * labse_query_embedding).sum())
            print("="*100)

