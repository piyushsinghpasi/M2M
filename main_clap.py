
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


from CLAP.model_clap import labse_clip
from CLAP.dataloader import TrainDataset, EmbeddingPrecomputator
from trainer_clap import Trainer

BASE_DIR = "/home/piyush/labse-clip/content/drive/MyDrive"


def print_metrics(m):
    # only printing r@10
    # order Avg     de      en      es      fr      it      jp      ko      pl      ru      tr      zh  

    keys = [k for k in m.keys() if "R@10" in k]
    lang_order = ['de', 'en', 'es', 'fr', 'it', 'jp', 'ko', 'pl', 'ru', 'tr', 'zh']
    # t2i
    keys_t2i = [k for k in keys if "t2i/" in k]
    lang_map = {k.split('_')[-1]: k for k in keys_t2i}

    metric_t2i = []
    for l in lang_order:
        col = lang_map[l]
        metric_t2i.append(m[col])

    avg_t2i = np.mean(metric_t2i).item()
    metric_t2i = [avg_t2i] + metric_t2i
    metric_t2i = [f"{x:.1f}" for x in metric_t2i]
    
    # i2t
    keys_i2t = [k for k in keys if "i2t/" in k]
    lang_map = {k.split('_')[-1]: k for k in keys_i2t}

    metric_i2t = []
    for l in lang_order:
        col = lang_map[l]
        metric_i2t.append(m[col])

    avg_i2t = np.mean(metric_i2t).item()
    metric_i2t = [avg_i2t] + metric_i2t
    metric_i2t = [f"{x:.1f}" for x in metric_i2t]


    print("Lang order")
    print("**".join(['Avg']+lang_order))

    print("T2I")
    print("**".join(metric_t2i))
    
    print("I2T")
    print("**".join(metric_i2t))

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
    parser.add_argument("--c_dist", type=int, default=48, 
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
            default=f"{BASE_DIR}/base-clip-data/models")
    parser.add_argument("--audio_model", type=str, help="image model clip/clap", 
            default="clip")
    parser.add_argument("--text_model", type=str, help="text model labse", 
            default="labse")
    parser.add_argument("--t2t_model1", type=str, help="model1 used to retrieve", 
            default="audio_text")
    parser.add_argument("--t2t_model2", type=str, help="model2 used to retrieve", 
            default="text")
    parser.add_argument('--eng_base_path', type=str, help="where all english txt files are stored",
            default=f"{BASE_DIR}/clip-data/AWS 68 Languages/Pre/")
    parser.add_argument('--non_eng_base_path', type=str, help="where all english txt files are stored",
            default=f"{BASE_DIR}/clip-data/AWS 68 Languages/Post/")
    parser.add_argument('--config', type=str, help="args in parser",
            default="")

    # I dont think eval_data_path is used for any but getting test or val name
    # for identifying split name- thats all
    # I think data is coming from map files
    # --img_to_text_map_path and --text_to_image_map_path
    parser.add_argument('--eval_data_path', type=str, help="eval csv metadata",
            default="/home/piyush/labse-clip/eval_data/audio_dataset/audiocaps/dataset/val.csv")
    parser.add_argument("--audio_base_dir", type=str, help="where all image/audio is saved", 
            default="/home/piyush/labse-clip/eval_data/audio_dataset/val/")
    parser.add_argument("--img_to_text_map_path", type=str, 
            help="where all image/audio is saved", 
    )
    parser.add_argument("--text_to_img_map_path", type=str, 
            help="where all image/audio is saved", 
    )

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        
        for key, value in config.items():
            if key in [
                    "mode", "eval_data_path", 
                    "audio_base_dir", "img_to_text_map_path", 
                    "text_to_img_map_path"
                ]:
                print(f"Skipping {key} from config")
            else:
                setattr(args, key, value)
    else:
        # add the args which are computed/don't come directly from config
        # during training
        args.save_dir = os.path.join(args.save_dir, args.wandb_name)
        
    wandb.init(project="Labse-CLIP", name=args.wandb_name, config=args)

    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Using {args.audio_model} and {args.text_model}")
    """# RUN

    ## Load model
    """
    if "clap_general" in args.audio_model.lower():
        from CLAP.model_clap_general import labse_clip
        print("Using clap-general class")

    device = "cuda"
    model = labse_clip(hdim=768, args=args)
    model = model.to(device)

    """## Load training data"""

    #langs = ['hi', 'ta', 'bn', 'te', 'ru', 'fr']
    langs = args.train_langs
    eng_file_paths = [os.path.join(args.eng_base_path, lang + '.txt') for lang in langs]
    #non_eng_file_paths = [os.path.join(args.non_eng_base_path, lang + '.txt') for lang in langs]
    non_eng_file_paths = []

    train_data = TrainDataset(
        eng_file_paths=eng_file_paths,
        non_eng_file_paths=non_eng_file_paths,
        labse_model=model.labse,
        clip_model=model.clip_model,
        clip_processor=model.clip_processor,
        clip_text_to_id_path = f"{BASE_DIR}/base-clip-data/{args.audio_model}_audio_text_to_id_train.pkl" ,
        labse_text_to_id_path = f"{BASE_DIR}/base-clip-data/{args.text_model}_audio_text_to_id_train.pkl",
        feature_save_dir = f"{BASE_DIR}/base-clip-data/features/audio/",
        audio_model = args.audio_model,
        text_model = args.text_model,
    )

    # implies 100 batches per epoch
    K = args.batch_size*args.num_batch

    if K < len(train_data.eng_data):
        print(f"Sampling {K} instances to train")
        
        random.seed(42)
        indices = random.sample(range(len(train_data.eng_data)), K)

        train_data.eng_data = [train_data.eng_data[i] for i in indices]
        if len(indices) <= len(train_data.non_eng_data):
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


    # below path is used to load and save text mappings
    text_to_id_path = f"{BASE_DIR}/base-clip-data/{args.text_model}_audio_text_to_id.pkl"


    os.makedirs(os.path.dirname(text_to_id_path), exist_ok=True)

    eval_dataset = EmbeddingPrecomputator(
        model.clip_model,
        model.clip_processor,
        model.labse,
        device,
        audio_dir = args.audio_base_dir,
        feature_save_dir = f"{BASE_DIR}/base-clip-data/features/audio/",
        img_to_text_map_path = args.img_to_text_map_path,
        text_to_img_map_path = args.text_to_img_map_path,
        text_to_id_path = text_to_id_path,
        audio_model = args.audio_model,
        text_model = args.text_model,
        data_split=args.eval_data_path.split('/')[-1].replace('.csv', ''),
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
            model.clip_model,
            model.clip_processor,
            model.labse,
            device,
            audio_dir = args.audio_base_dir,
            feature_save_dir = f"{BASE_DIR}/base-clip-data/features/audio/",
            img_to_text_map_path = args.img_to_text_map_path,
            text_to_img_map_path = args.text_to_img_map_path,
            text_to_id_path = text_to_id_path,
            audio_model = args.audio_model,
            text_model = args.text_model,
            data_split=args.eval_data_path.split('/')[-1].replace('.csv', ''),
        )

        trainer.eval_dataset = eval_dataset
        metrics = trainer.evaluate(labse_baseline=True)
        with open(f"./metrics/baseline_{args.text_model}.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved at: ./metrics/baseline_{args.text_model}.json")
    elif "baseline_clap" in args.mode:
        eval_dataset = EmbeddingPrecomputator(
            model.clip_model,
            model.clip_processor,
            model.labse,
            device,
            audio_dir = args.audio_base_dir,
            feature_save_dir = f"{BASE_DIR}/base-clip-data/features/audio/",
            img_to_text_map_path = args.img_to_text_map_path,
            text_to_img_map_path = args.text_to_img_map_path,
            text_to_id_path = text_to_id_path,
            audio_model = args.audio_model,
            text_model = args.text_model,
            data_split=args.eval_data_path.split('/')[-1].replace('.csv', ''),
        )
        trainer.eval_dataset = eval_dataset
        metrics = trainer.evaluate(use_clip=True)
        if args.save_ranks:
            metrics, ranks = metrics
            with open(f"./ranks/{args.wandb_name}_{args.text_model}_{args.mode}.pkl", "wb") as f:
                pickle.dump(ranks, f)
            print(f"Ranks saved at: ./ranks/{args.wandb_name}_{args.text_model}_{args.mode}.pkl")
        
        with open(f"./metrics/{args.wandb_name}_{args.text_model}_{args.mode}.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved at: ./metrics/{args.wandb_name}_{args.text_model}_{args.mode}.json")
    
    elif "eval_labse" in args.mode:
        
        save_path = os.path.join(args.save_dir, "checkpoint.pth")
        loaded_layers = torch.load(save_path, map_location='cpu')
        model.load_state_dict(loaded_layers, strict=False)
        
        eval_dataset = EmbeddingPrecomputator(
            model.clip_model,
            model.clip_processor,
            model.labse,
            device,
            audio_dir = args.audio_base_dir,
            feature_save_dir = f"{BASE_DIR}/base-clip-data/features/audio/",
            img_to_text_map_path = args.img_to_text_map_path,
            text_to_img_map_path = args.text_to_img_map_path,
            text_to_id_path = text_to_id_path,
            audio_model = args.audio_model,
            text_model = args.text_model,
            data_split=args.eval_data_path.split('/')[-1].replace('.csv', ''),
        )

        data_split=args.eval_data_path.split('/')[-1].replace('.csv', '')
        trainer.eval_dataset = eval_dataset
        metrics = trainer.evaluate(use_clip=False)
        if args.save_ranks:
            metrics, ranks = metrics
            with open(f"./ranks/{args.wandb_name}_{args.text_model}_{args.mode}_split-{data_split}.pkl", "wb") as f:
                pickle.dump(ranks, f)
            print(f"Ranks saved at: ./ranks/{args.wandb_name}_{args.text_model}_{args.mode}_split-{data_split}.pkl")
        
        with open(f"./metrics/{args.wandb_name}_{args.text_model}_{args.mode}_split-{data_split}.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved at: ./metrics/{args.wandb_name}_{args.text_model}_{args.mode}_split-{data_split}.json")

        #print("R@10 metrics")
        #print_metrics(metrics)

    elif args.mode == "compare_embs":

        
        save_path = os.path.join(args.save_dir, "checkpoint.pth")
        loaded_layers = torch.load(save_path, map_location='cpu')
        model.load_state_dict(loaded_layers, strict=False)
        
        eval_dataset = EmbeddingPrecomputator(
            aud_text_df,
            model.clip_model,
            model.clip_processor,
            model.labse,
            device,
            audio_dir = f"{BASE_DIR}/base-clip-data/audios",
            feature_save_dir = f"{BASE_DIR}/base-clip-data/features",
            lang_cols = ['XTD10_captions_it',
               'XTD10_captions_ru', 'caption.', 'XTD10_captions_pl',
               'XTD10_captions_tr', 'XTD10_captions_zh', 'XTD10_captions_ko',
               'XTD10_captions_es', 'STAIR_caption_jp', 'MIC_caption_de',
               'MIC_caption_fr'],
            aud_col = 'test_audio_names',
            text_to_id_path = text_to_id_path,
            audio_model = args.audio_model,
            text_model = args.text_model,
            data_split=args.eval_data_path.split('/')[-1].replace('.csv', ''),
        )


        trainer.eval_dataset = eval_dataset
        
        aud_text_df = aud_text_df.head(5)
        lang = 'caption'
        
        for idx, row in aud_text_df.iterrows():

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

    elif args.mode == "eval_t2t":
        skip_inference=False
        
        save_path = os.path.join(args.save_dir, "checkpoint.pth")
        if not os.path.exists(save_path):
            x = int(input("Model load path doesn't exist, press 1 if its baseline"))
            if x!=1:
                raise ValueError("Model savepath doesnt exist")
            skip_inference = True
        else:
            loaded_layers = torch.load(save_path, map_location='cpu')
            model.load_state_dict(loaded_layers, strict=False)
        
        eval_dataset = EmbeddingPrecomputator(
            aud_text_df,
            model.clip_model,
            model.clip_processor,
            model.labse,
            device,
            audio_dir = f"{BASE_DIR}/base-clip-data/audios",
            feature_save_dir = f"{BASE_DIR}/base-clip-data/features",
            lang_cols = ['XTD10_captions_it',
               'XTD10_captions_ru', 'caption.', 'XTD10_captions_pl',
               'XTD10_captions_tr', 'XTD10_captions_zh', 'XTD10_captions_ko',
               'XTD10_captions_es', 'STAIR_caption_jp', 'MIC_caption_de',
               'MIC_caption_fr'],
            aud_col = 'test_audio_names',
            text_to_id_path = text_to_id_path,
            audio_model = args.audio_model,
            text_model = args.text_model,
            data_split=args.eval_data_path.split('/')[-1].replace('.csv', ''),
        )

        t2t_order = {
                'model1': args.t2t_model1,
                'model2': args.t2t_model2,
        }

        if args.t2t_model2=='audio_text':
            skip_inference=True

        trainer.eval_dataset = eval_dataset
        metrics = trainer.evaluate(return_top_idx=args.save_ranks, t2t=True, t2t_order=t2t_order,skip_inference=skip_inference)
        save_filename = f"t2t_{args.wandb_name}_{args.t2t_model1}_to_{args.t2t_model2}"

        if args.save_ranks:
            metrics, ranks = metrics
            with open(f"./ranks/{save_filename}.pkl", "wb") as f:
                pickle.dump(ranks, f)
            print(f"Ranks saved at: ./ranks/{save_filename}.pkl")
        
        with open(f"./metrics/{save_filename}.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved at: ./metrics/{save_filename}.json")

