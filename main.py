
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel, AutoTokenizer, AutoModel
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

from PIL import Image

from model import labse_clip
from dataloader import TrainDataset, EmbeddingPrecomputator
from trainer import Trainer

from recall_k_metric import retrieval_evaluation
#from recall_k_metric_text_redo import retrieval_evaluation
from functools import partial

from multilingual_clip import pt_multilingual_clip
import open_clip
import clip


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


def crossmodal3600_print(all_metrics, dataset_name=""):
    
    i2t_r_10 = []
    t2i_r_10 = []

    langs = all_metrics.keys()
    
    if dataset_name == "XTD10":
        langs = [(l.split('_')[-1], l) for l in langs]
    else:
        langs = [(l, l) for l in langs]

    langs = sorted(langs, key=lambda x: x[0])
    
    for l_code, l_col in langs:
        metrics = all_metrics[l_col]
        i2t_r_10.append(metrics["image_to_text"]["recall@10"])
        t2i_r_10.append(metrics["text_to_image"]["recall@10"])
            
    avg_i2t = np.mean(i2t_r_10).item()
    avg_t2i = np.mean(t2i_r_10).item()

    
    print("Lang order")
    print("**".join(['Avg']+[l_code for l_code,_ in langs]))

    print("T2I")
    t2i = [avg_t2i] + t2i_r_10
    t2i = [f"{x:.1f}" for x in t2i]
    print("**".join(t2i))
    
    print("I2T")
    i2t = [avg_i2t] + i2t_r_10
    i2t = [f"{x:.1f}" for x in i2t]
    print("**".join(i2t))

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
    parser.add_argument("--c_dist", type=float, default=44, 
            help="support diff losses see trainer.py, "
            "used for linear comb. wt. for mse/l1")
    parser.add_argument("--c_kl", type=float, default=1, 
            help="support diff losses see trainer.py, "
            "used for linear comb. wt. for kl")
    parser.add_argument("--c_kl_clip", type=float, default=1, 
            help="support diff losses see trainer.py, "
            "used for linear comb. wt. for kl clip")
    parser.add_argument("--c_eng_cosine", type=int, default=1, 
            help="support diff losses see trainer.py, "
            "used for linear comb. wt. for kl clip")
    parser.add_argument("--topk", type=int, default=5, help="used for topk in L_str loss; topk on GT sim")
    parser.add_argument("--early_stopping", action="store_true", help="earlystopping")
    parser.add_argument("--save_ranks", action="store_true", help="save predicted ranks")
    parser.add_argument("--save_feats", action="store_true", help="save predicted/aligned features")
    parser.add_argument("--penalty_gap", type=int, default=10, 
            help="epochs to wait after best epoch before triggering early stopping")
    parser.add_argument("--save_dir", type=str, help="dir where checkpoint", 
            default="./content/drive/MyDrive/base-clip-data/models")
    parser.add_argument("--image_model", type=str, help="image model clip/jinav1", 
            default="clip")
    parser.add_argument("--text_model", type=str, help="text model labse", 
            default="labse")
    parser.add_argument("--t2t_model1", type=str, help="model1 used to retrieve", 
            default="image_text")
    parser.add_argument("--t2t_model2", type=str, help="model2 used to retrieve", 
            default="text")
    parser.add_argument('--eng_base_path', type=str, help="where all english txt files are stored",
            default="./content/drive/MyDrive/clip-data/AWS 68 Languages/Pre/")
    parser.add_argument('--non_eng_base_path', type=str, help="where all english txt files are stored",
            default="./content/drive/MyDrive/clip-data/AWS 68 Languages/Post/")
    parser.add_argument('--config', type=str, help="args in parser",
            default="")
    parser.add_argument("--base_image_dir", type=str, help="dir path to images",
            default="./content/drive/MyDrive/base-clip-data/images")
    parser.add_argument("--base_feature_dir", type=str, help="dir path to features",
            default="./content/drive/MyDrive/base-clip-data/features")

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
        
        for key, value in config.items():
            if key == "mode":
                print(f"Skipping mode from config, mode- {args.mode} will be used")
            else:
                setattr(args, key, value)
    else:
        # add the args which are computed/don't come directly from config
        # during training
        args.save_dir = os.path.join(args.save_dir, args.wandb_name)
        
    wandb.init(project="Labse-CLIP", name=args.wandb_name, config=args)

    os.makedirs(args.save_dir, exist_ok=True)

    hdim = 768
    if args.emb_method == "hybrid":
        from model_hybrid import labse_clip
    if args.image_model == "jinav1" and args.text_model == "labse":
        from model_jinav1 import labse_clip
        print("model jinav1 and labse")
    elif args.image_model == "jinav1" and args.text_model == "multiMiniLM":
        from model_jinav1_multiMiniLM import labse_clip
        print("model_jinav1_multiMiniLM loaded")
    elif args.image_model == "jinav1" and args.text_model == "multiMpnet":
        if args.emb_method == 'gaussian':
            from model_jinav1_multiMpnet_gaussian import labse_clip
            print("model_jinav1_multiMpnet_gaussian")
        elif args.emb_method == 'topk':
            from model_jinav1_multiMpnet_topk import labse_clip
            print("model_jinav1_multiMpnet_topk")
        elif args.emb_method == 'toplastk':
            from model_jinav1_multiMpnet_toplastk import labse_clip
            print("model_jinav1_multiMpnet_toplastk")
        else:
            from model_jinav1_multiMpnet import labse_clip
            print("model_jinav1_multiMpnet")
    elif args.image_model == "siglip2" and args.text_model ==  "multiMpnet":
        from model_siglip2_multiMpnet import labse_clip
        print("loading model_siglip2_multiMpnet")
    elif args.image_model == "siglip2_so400" and args.text_model ==  "multiMpnet":
        from model_siglip2_so400_multiMpnet import labse_clip
        print("loading model_siglip2_so400_multiMpnet.py")
    elif args.image_model == "siglip1" and args.text_model ==  "multiMpnet":
        from model_siglip1_multiMpnet import labse_clip
        print("loading model_siglip1_multiMpnet.py")
        hdim = 1152
    elif args.image_model == "siglip2_base" and args.text_model ==  "multiMpnet":
        from model_siglip2_base_multiMpnet import labse_clip
        print("loading model_siglip2_multiMpnet")
    elif args.image_model == "jinav1" and args.text_model == "jinaTextv3":
        from model_jinav1_jinaTextv3 import labse_clip
        print("model_jinav1_jinaTextv3")
    elif args.image_model == "clip" and args.text_model == "multiMpnet":
        from model_clip_multiMpnet import labse_clip
        print("model_clip_multiMpnet")
    elif args.image_model == "align" and args.text_model == "multiMpnet":
        from model_align_multiMpnet import labse_clip
        print("model_align_multiMpnet")

    print(f"Using {args.image_model} and {args.text_model}")
    """# RUN

    ## Load model
    """

    device = "cuda"
    model = labse_clip(hdim=hdim, args=args)
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
        clip_text_to_id_path = f"./content/drive/MyDrive/base-clip-data/{args.image_model}_text_to_id_train.pkl" ,
        labse_text_to_id_path = f"./content/drive/MyDrive/base-clip-data/{args.text_model}_text_to_id_train.pkl",
        feature_save_dir = args.base_feature_dir,
        image_model = args.image_model,
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


    eval_data_path = "./content/drive/MyDrive/base-clip-data/Cross-lingual-Test-Dataset-XTD10/merged_data.csv"
    img_text_df = pd.read_csv(eval_data_path)

    # below path is used to load and save text mappings
    text_to_id_path = "./content/drive/MyDrive/base-clip-data/text_to_id.pkl"
    if args.text_model != "labse":
        text_to_id_path = f"./content/drive/MyDrive/base-clip-data/{args.text_model}_text_to_id.pkl"


    os.makedirs(os.path.dirname(text_to_id_path), exist_ok=True)

    eval_dataset = EmbeddingPrecomputator(
        img_text_df,
        model.clip_model,
        model.clip_processor,
        model.labse,
        device,
        image_dir = args.base_image_dir,
        feature_save_dir = args.base_feature_dir,
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
            image_dir = args.base_image_dir,
            feature_save_dir = args.base_feature_dir,
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
            image_dir = args.base_image_dir,
            feature_save_dir = args.base_feature_dir,
            #lang_cols = ['XTD10_captions_en'],
            lang_cols = [
               'XTD10_captions_it',
               'XTD10_captions_ru', 'XTD10_captions_en', 'XTD10_captions_pl',
               'XTD10_captions_tr', 'XTD10_captions_zh', 'XTD10_captions_ko',
               'XTD10_captions_es', 
               'MIC_caption_de',
               'STAIR_caption_jp', 
               'MIC_caption_fr'],
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
        
        print("R@10 metrics")
        print_metrics(metrics)
    
    elif args.mode == "eval_labse":
        
        save_path = os.path.join(args.save_dir, "checkpoint.pth")
        loaded_layers = torch.load(save_path, map_location='cpu')
        model.load_state_dict(loaded_layers, strict=False)
        
       
        eval_dataset = EmbeddingPrecomputator(
            img_text_df,
            model.clip_model,
            model.clip_processor,
            model.labse,
            device,
            image_dir = args.base_image_dir,
            feature_save_dir = args.base_feature_dir,
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
        
        if args.save_feats:
            feat_path = os.path.join(args.save_dir, "features")
            print(f"Features saved at : {feat_path}")
        
        if args.save_ranks:
            metrics, ranks = metrics
            with open(f"./ranks/{args.wandb_name}_{args.text_model}.pkl", "wb") as f:
                pickle.dump(ranks, f)
            print(f"Ranks saved at: ./ranks/{args.wandb_name}_{args.text_model}.pkl")
        
        with open(f"./metrics/{args.wandb_name}_{args.text_model}.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved at: ./metrics/{args.wandb_name}_{args.text_model}.json")

        print("R@10 metrics")
        print_metrics(metrics)

    elif args.mode == "eval_categories":
        
        save_path = os.path.join(args.save_dir, "checkpoint.pth")
        loaded_layers = torch.load(save_path, map_location='cpu')
        model.load_state_dict(loaded_layers, strict=False)
        
        eval_data_path = "./Synthetic_Multilingual_Eval_Dataset/MSCOCO-30K-10-lang.csv" 
        img_text_df_all = pd.read_csv(eval_data_path) 

        with open("analysis_data_and_scripts/object_coco_data.pkl", "rb") as f:
            object_coco_data = pickle.load(f)
        
        for category, rows  in object_coco_data.items():
            idxs = [row['id'] for row in rows]
            
            img_text_df = img_text_df_all.iloc[idxs]
            print(f"{category} {len(idxs)} {img_text_df.shape}")
            
            eval_data_filename = eval_data_path.split('/')[-1][:-4]

            # only base features are saved here; these may not be aligned with order of the csv file
            custom_base_feat_dir = os.path.join(args.save_dir, "eval_categories_base_features", eval_data_filename)
            os.makedirs(custom_base_feat_dir, exist_ok=True)

            # all types of features are save here (base, aligned, etc.)
            # in order of the csv file
            feat_path = os.path.join(args.save_dir, category)

            # remove old feats
            # note base feats need not be removed
            #if os.path.exists(feat_path):
            #    os.system(f"rm -rf {feat_path}/*")

            text_to_id_path = os.path.join(args.save_dir, eval_data_filename+".pkl")

            eval_dataset = EmbeddingPrecomputator(
                img_text_df,
                model.clip_model,
                model.clip_processor,
                model.labse,
                device,
                image_dir = args.base_image_dir,
                feature_save_dir = custom_base_feat_dir,
                lang_cols = ['caption', 'French', 'Greek', 'Hebrew', 'Indonesian',
                           'Korean', 'Persian', 'Russian', 'Spanish', 'Hindi'],
                img_col = 'file_name',
                text_to_id_path = text_to_id_path,
                image_model = args.image_model,
                text_model = args.text_model,
            )


            trainer.eval_dataset = eval_dataset
            metrics = trainer.evaluate(return_top_idx=args.save_ranks, save_feat_dir = feat_path)
            
            if args.save_feats:
                print(f"Features saved at : {feat_path}")
            
            if args.save_ranks:
                metrics, ranks = metrics
                with open(f"./ranks/{args.wandb_name}_eval_categories.pkl", "wb") as f:
                    pickle.dump(ranks, f)
                print(f"Ranks saved at: ./ranks/{args.wandb_name}_eval_categories.pkl")
            
            with open(f"./metrics/{args.wandb_name}_eval_categories.json", "w") as f:
                json.dump(metrics, f, indent=4)
            print(f"Metrics saved at: ./metrics/{args.wandb_name}_eval_categories.json")

            #print("R@10 metrics")
            #print_metrics(metrics)


            with open(text_to_id_path, 'wb') as f:
                pickle.dump(eval_dataset.text_to_id, f)
            
            print(f"Text to ID len: {len(eval_dataset.text_to_id)}")
            print("Saved Text to ID map")

        
        
    elif args.mode == "compare_embs":

        
        save_path = os.path.join(args.save_dir, "checkpoint.pth")
        loaded_layers = torch.load(save_path, map_location='cpu')
        model.load_state_dict(loaded_layers, strict=False)
        
        eval_dataset = EmbeddingPrecomputator(
            img_text_df,
            model.clip_model,
            model.clip_processor,
            model.labse,
            device,
            image_dir = args.base_image_dir,
            feature_save_dir = args.base_feature_dir,
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
            img_text_df,
            model.clip_model,
            model.clip_processor,
            model.labse,
            device,
            image_dir = args.base_image_dir,
            feature_save_dir = args.base_feature_dir,
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

        t2t_order = {
                'model1': args.t2t_model1,
                'model2': args.t2t_model2,
        }

        if args.t2t_model2=='image_text':
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

    
    elif "crossmodal3600" in args.mode or "M30K" in args.mode:
        # general recall function
        # needs two dictionaries 
        # 1. Image as Key and list of text as value (Ground truth)
        # 2. Text as Key and list of images as value (Ground truth)
        
        img_ext = ""
        if "crossmodal3600" in args.mode:
            print("CrossModal3600")

            img_to_text_map_path = "eval_data/crossmodal3600/image_to_text_mapping.pkl"
            with open(img_to_text_map_path, "rb") as f:
                img_to_text_map = pickle.load(f)
            
            text_to_img_map_path = "eval_data/crossmodal3600/text_to_image_mapping.pkl"
            with open(text_to_img_map_path, "rb") as f:
                text_to_img_map = pickle.load(f)
            
            # overwriting these as config has dev-set paths used during training
            custom_base_image_dir = "./eval_data/crossmodal3600/images/" 
            custom_base_feature_dir = os.path.join(args.base_feature_dir, "crossmodal3600-2")
        
            img_ext = ".jpg"
        else:
            print("Multi30K M30K")
            img_to_text_map_path = "eval_data/multi30k-dataset/image_to_text_mapping.pkl"
            with open(img_to_text_map_path, "rb") as f:
                img_to_text_map = pickle.load(f)
            
            text_to_img_map_path = "eval_data/multi30k-dataset/text_to_image_mapping.pkl"
            with open(text_to_img_map_path, "rb") as f:
                text_to_img_map = pickle.load(f)
        
            # overwriting these as config has dev-set paths used during training
            custom_base_image_dir = "./eval_data/multi30k-dataset/flickr30k/Images/" 
            custom_base_feature_dir = os.path.join(args.base_feature_dir, "multi30k-2")
        
        use_clip  = False
        # load model
        if "baseline" not in args.mode:
            save_path = os.path.join(args.save_dir, "checkpoint.pth")
            loaded_layers = torch.load(save_path, map_location='cpu')

            model.load_state_dict(loaded_layers, strict=False)
            
            use_clip = False
            print("Model Loaded!!")
        else:
            use_clip = True
            print("Running baseline")

        model.eval()

        text_to_id_path = f"./content/drive/MyDrive/base-clip-data/{args.text_model}_text_to_id_evals.pkl"

        # create eval dataset        
        eval_dataset = EmbeddingPrecomputator(
            None,
            model.clip_model,
            model.clip_processor,
            model.labse,
            device,
            image_dir = custom_base_image_dir,
            feature_save_dir = custom_base_feature_dir,
            lang_cols = None,
            img_col = None,
            text_to_id_path = text_to_id_path,
            image_model = args.image_model,
            text_model = args.text_model,
        )

        langs = sorted(img_to_text_map.keys())

        all_metrics = dict()
        with torch.no_grad():
            for l in langs:
                print(f"Evaluating {l}")
                #if l != 'en': continue
                
                get_text_feature = partial(eval_dataset.get_text_embedding, lang=l, use_clip=use_clip)
                wrapper_func = get_text_feature
                wrapper_func_baseline = get_text_feature  
                if "baseline" not in args.mode:
                    def get_text_embed(text, get_t_feat):
                        t_feat = get_t_feat(text).to(device)
                        t_feat = model.inference(t_feat).detach().cpu()
                        return t_feat
                    wrapper_func = partial(get_text_embed, get_t_feat=get_text_feature)

                l_metric = retrieval_evaluation(
                    img_to_text_map[l], 
                    text_to_img_map[l], 
                    get_image_feature=eval_dataset.get_image_embedding,
                    get_text_feature=wrapper_func,
                    img_ext = img_ext ,
                    
                    # used only for text comparison (text recall eval file)
                    # get_text_base = wrapper_func_baseline,
                )

                all_metrics[l] = l_metric

                print(l_metric)
                print("="*100)

        crossmodal3600_print(all_metrics)
        
        save_filename = args.mode
        with open(f"./metrics/{save_filename}.json", "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Metrics saved at: ./metrics/{save_filename}.json")


        # save text mappings
        with open(text_to_id_path, "wb") as f:
            pickle.dump(eval_dataset.text_to_id, f)

    elif "Other" in args.mode:
        # to run inference on other models for baseline

        SPACER = "@@@"
        cos_util = False
        img_ext = ''
        image_base_dir = ""
       
        # XTD10
        if "XM3600" not in args.mode and "multi30k" not in args.mode:

            lang_cols = [('it', 'XTD10_captions_it'),
               ('ru', 'XTD10_captions_ru'), ('en', 'XTD10_captions_en'), ('pl', 'XTD10_captions_pl'),
               ('tr', 'XTD10_captions_tr'), ('zh', 'XTD10_captions_zh'), ('ko', 'XTD10_captions_ko'),
               ('es', 'XTD10_captions_es'), ('jp', 'STAIR_caption_jp'), ('de', 'MIC_caption_de'),
               ('fr', 'MIC_caption_fr')]
            img_col = 'test_image_names'
            
            img_to_text_map = {l: dict() for l,_ in lang_cols}
            text_to_img_map = {l: dict() for l,_ in lang_cols}
            
            imgs = img_text_df[img_col].tolist()
            
             
            for l, l_col in lang_cols:
                print(f"Mapping {l}")
                texts = img_text_df[l_col].tolist()
                
                text_id = [f'caption_{i_}' for i_ in range(len(texts))]
                text_id_map = {k:v  for k,v in zip(text_id, texts)}

                for k,v in zip(imgs, text_id):
                    k = os.path.join(args.base_image_dir, k)
                    
                    text_to_img_map[l][v] = [k]
                    
                    img_to_text_map[l][k] = [v]

                text_to_img_map[l]["unique_text_id_to_caption_map"] = text_id_map

        elif "multi30k" in args.mode:
            # multi30k
            img_to_text_map_path = "eval_data/multi30k-dataset/image_to_text_mapping.pkl"
            with open(img_to_text_map_path, "rb") as f:
                img_to_text_map = pickle.load(f)
            
            text_to_img_map_path = "eval_data/multi30k-dataset/text_to_image_mapping.pkl"
            with open(text_to_img_map_path, "rb") as f:
                text_to_img_map = pickle.load(f)
            
            lang_cols = sorted(img_to_text_map.keys())
            
            image_base_dir = ""
        
        else:
            # XM3600
            img_to_text_map_path = "eval_data/crossmodal3600/image_to_text_mapping.pkl"
            with open(img_to_text_map_path, "rb") as f:
                img_to_text_map = pickle.load(f)
            
            text_to_img_map_path = "eval_data/crossmodal3600/text_to_image_mapping.pkl"
            with open(text_to_img_map_path, "rb") as f:
                text_to_img_map = pickle.load(f)
            
            lang_cols = sorted(img_to_text_map.keys())
            img_ext = ".jpg"

            image_base_dir = "./eval_data/crossmodal3600/images/"
        
    
        if "M-CLIP" in args.mode:
            # load model
            if "M-CLIP/LABSE-Vit-L-14" in args.mode:
                text_model_name = 'M-CLIP/LABSE-Vit-L-14'
                model_image, preprocess = clip.load("ViT-L/14", device=device)

            elif "M-CLIP/XLM-Roberta-Large-Vit-B-32" in args.mode:
                text_model_name = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
                model_image, preprocess = clip.load("ViT-B/32", device=device)

            elif "M-CLIP/XLM-Roberta-Large-Vit-L-14" in args.mode:
                text_model_name = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
                model_image, preprocess = clip.load("ViT-L/14", device=device)

            elif "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus" in args.mode:
                text_model_name = "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus"
                model_image, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")


            # Load Model & Tokenizer
            model_text = pt_multilingual_clip.MultilingualCLIP.from_pretrained(text_model_name)
            tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            model_text.to(device)
            model_text.eval()

        
            model_image.to(device)
            model_image.eval()
            
            
            @torch.no_grad()
            def get_text_feature_(text, model, tokenizer, spacer, device):
                # remove SPACER
                text = text.split(spacer)[0]
                embeddings = model.forward([text], tokenizer, device)
                return embeddings

            get_text_feature = partial(get_text_feature_, model=model_text, tokenizer=tokenizer, spacer=SPACER, device=device)


            @torch.no_grad()
            def get_image_feature_(image_path, model, preprocessor, device):
                image = Image.open(image_path)
                image = preprocess(image).unsqueeze(0).to(device)
                image_features = model.encode_image(image).float()
                return image_features
            
            get_image_feature = partial(get_image_feature_, model=model_image, preprocessor=preprocess, device=device)

        elif "jina" in args.mode:
            model = AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)
            model.to(device)
            model.eval()

    
            @torch.no_grad()
            def get_text_feature_(text, model, spacer):
                text = text.split(spacer)[0]
                text_embeddings = model.encode_text(
                        [text], 
                        #task = "text-matching",
                )
                text_embeddings = torch.from_numpy(text_embeddings)
                return text_embeddings
    
            @torch.no_grad()
            def get_image_feature_(image_path, model):
                img_feat = model.encode_image([image_path])
                img_feat = torch.from_numpy(img_feat)
                return img_feat

            get_text_feature = partial(get_text_feature_, model=model, spacer=SPACER)
            get_image_feature = partial(get_image_feature_, model=model)

        
        elif "sentCLIP" in args.mode:
            print(f"Mode: {args.mode}")

            img_model = SentenceTransformer('clip-ViT-B-32')
            text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
            
            img_model.to(device)
            text_model.to(device)

            img_model.eval()
            text_model.eval()

            @torch.no_grad()
            def get_text_feature_(text, model, spacer, device=device):
                text = text.split(spacer)[0]
                text_embeddings = model.encode([text], device=device)
                text_embeddings = torch.from_numpy(text_embeddings)
                return text_embeddings
    
            @torch.no_grad()
            def get_image_feature_(image_path, model, device):
                image = Image.open(image_path).convert("RGB") 
                img_feat = model.encode([image], device=device)
                img_feat = torch.from_numpy(img_feat)
                return img_feat

            get_text_feature = partial(get_text_feature_, model=text_model, spacer=SPACER, device=device)
            get_image_feature = partial(get_image_feature_, model=img_model, device=device)
        

            cos_util = True
        
        all_metrics = dict()
        prev = None
        precomputed_feats = dict()
        with torch.no_grad():
            for l, l_col in lang_cols:
                print(f"Evaluating {l}")
                image_to_captions = img_to_text_map[l]
                text_to_images = text_to_img_map[l]
                
                images = list(image_to_captions.keys())
                images = sorted(images)
                
                # each key here is text-ID, value is [img, caption]
                # gauranteed to be 1:1 map
                text_id_to_caption_map = dict()
                
                if "unique_text_id_to_caption_map" in text_to_images:
                    text_id_to_caption_map = text_to_images.pop("unique_text_id_to_caption_map")
                print("text_id_to_caption_map", len(text_id_to_caption_map))
                
                captions = list(text_to_images.keys())

                # put unique ids back in mappings, for subsequent epochs
                text_to_images["unique_text_id_to_caption_map"] =  text_id_to_caption_map

                # Get features 
                # T x D
                texts = [text_id_to_caption_map.get(caption, caption) for caption in captions]
                print(texts[:5])
                text_features = torch.cat([
                    get_text_feature(text_id_to_caption_map.get(caption, caption)) 
                    for caption in captions
                ], dim=0)
                
                #text_features = torch.cat([get_text_feature(caption) for caption in captions], dim=0)
                print("text", text_features.shape)
                
                
                if "image_features" not in precomputed_feats:
                    # IxD
                    # one time compute image features
                    if image_base_dir == "":
                        image_features = torch.cat([get_image_feature(img+img_ext) for img in images], dim=0)
                    else:
                        image_features = torch.cat([
                            get_image_feature(
                                os.path.join(image_base_dir, img+img_ext)
                            ) 
                            for img in images
                        ], dim=0)

                    precomputed_feats["image_features"] = image_features
                    precomputed_feats["images"] = images
                else:
                    print("using old image features")
                # always overwrite text attrs
                precomputed_feats["captions"] = captions
                precomputed_feats["text_features"] = text_features
            
                l_metric = retrieval_evaluation(
                    img_to_text_map[l], 
                    text_to_img_map[l], 
                    get_image_feature,
                    get_text_feature,
                    cos_util=cos_util,
                    precomputed_features = precomputed_feats,
                )
                
                print(l_metric)
                all_metrics[l] = l_metric
                print("="*100)

        #crossmodal3600_print(all_metrics, dataset_name="XTD10")
        crossmodal3600_print(all_metrics)
        
        save_filename = args.mode+".json"
        save_p = os.path.join("./metrics", save_filename)

        os.makedirs(os.path.dirname(save_p), exist_ok=True)
        with open(save_p, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Metrics saved at: {save_p}")
