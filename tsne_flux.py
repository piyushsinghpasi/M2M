import torch
from typing import Union, List, Optional

from flux.model import labse_clip
#from model_simple import labse_clip

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, CLIPTextModel, CLIPVisionModel
from argparse import ArgumentParser

import os
from functools import partial 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import string
import time
import pandas as pd

from tqdm import tqdm
import json

def remove_punc(s):
    return s.translate(str.maketrans('', '', string.punctuation))

def clean_text(x):
    x = str(x)
    x = x.lower()
    x = remove_punc(x)
    x = ' '.join(x.split())
    x = x.replace(' ', '_')
    return x



def save_image_grid(image_paths, captions, txt, output_path):
    # Create a figure with a grid layout
    fig, axes = plt.subplots(2,3, figsize=(12, 8))
    fig.suptitle(txt, fontsize=16, fontweight="bold", y=0.95)

    # Adjust spacing
    plt.subplots_adjust(hspace=0.5, wspace=0.3)

    # Add images and captions
    for i, ax in enumerate(axes.flat):
        img = mpimg.imread(image_paths[i])

        ax.imshow(img, aspect='auto')
        ax.set_title(captions[i], fontsize=12)
        ax.axis("off")  # Hide axes

    # Show the grid
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Grid saved as {output_path}")

#LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
LABSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

parser = ArgumentParser()
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--epoch", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--emb_method", type=str, default="no_skip_conn", 
    help="used in model for skip conn or sequential transformation")
parser.add_argument("--wandb_name", type=str, help="wandb run name")
parser.add_argument("--prompt_file", type=str, help="wandb run name")
parser.add_argument("--save_dir", type=str, help="dir where checkpoint", 
        
        default="/home/piyush/labse-clip/content/drive/MyDrive/base-clip-data/models"
)
parser.add_argument("--image_save_dir", type=str, help="generated image save dir", 
        default="./generated_images/")
parser.add_argument("--csv_file", type=str, 
        default="path to csv file which contains captions and image names")
parser.add_argument("--text_col", type=str, 
        default='captions column name in df')
parser.add_argument("--method", type=str,
        help='runs clip_only, baseline (same prompt2 as prompt), or labse')
parser.add_argument("--nhead", type=int, default=-1, help="pick top n utterance, -1 means pick all")
parser.add_argument("--prompt_2", type=str, 
        default="A photo of: ", 
        help="prompt used for T5 encoder"
    )
args = parser.parse_args()


LABSE_MODEL_NAME = ""
if "labse" in args.wandb_name.lower():
    print("LABSE used")
    LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
else:
    print("MPNET USED")
    LABSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

DEVICE='cuda'
#DEVICE='cpu'

def load_labse_model():
    model = labse_clip(hdim=768, args=args)

    labse = SentenceTransformer(LABSE_MODEL_NAME, device=DEVICE)
    if "labse" in LABSE_MODEL_NAME.lower():
        labse[3] = torch.nn.Identity()
    labse = labse.to(DEVICE)

    save_path = os.path.join(args.save_dir, args.wandb_name, f"epoch_{args.epoch}.pth")
    loaded_layers = torch.load(save_path, map_location='cpu')
    model.load_state_dict(loaded_layers, strict=False)

    print(loaded_layers.keys())
    model = model.to(DEVICE).to(dtype=torch.bfloat16)
    #model = model.to(dtype=torch.bfloat16)
    model.eval()
    return model, labse

@torch.no_grad()
def get_labse_emb(
    custom_model = None,
    custom_labse = None,
    text = None,
    ):
   
    text_emb = torch.from_numpy(custom_labse.encode([text])).to(DEVICE).to(dtype=torch.bfloat16)
    #text_emb = torch.from_numpy(custom_labse.encode(prompt)).to(dtype=torch.bfloat16)
    org_text_emb = text_emb
    text_emb = custom_model.inference(text_emb)

    text_emb = text_emb.detach().cpu().float().numpy()
    
    return text_emb, org_text_emb
        
def load_clip():

    CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, torch_dtype=torch.bfloat16).text_model
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    return clip_model, clip_processor

@torch.no_grad()
def get_clip_emb(clip_model, clip_processor, eng_text):

    inputs = clip_processor(
        text=[eng_text], 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
          inputs[k] = v.to(DEVICE)

    eng_clip_emb = clip_model(**inputs).pooler_output.detach().cpu()
    eng_clip_emb = eng_clip_emb.float().numpy()
    
    return eng_clip_emb

model, labse = load_labse_model()
clip_model, clip_processor = load_clip()


clip_model = clip_model.to(DEVICE)
model = model.to(DEVICE)
labse = labse.to(DEVICE)

clip_model.eval()
model.eval()
labse.eval()


df = pd.read_csv("eval_data/coco2014/MSCOCO-30K-10-lang.csv") 
df = df.rename(columns={"caption": "English"})

#df = df.head(1000)

langs = ['English', 'French', 'Greek', 'Hebrew', 'Indonesian',
               'Korean', 'Persian', 'Russian', 'Spanish', 'Hindi']

for idx, row in tqdm(df.iterrows(), desc="extracting"):
    
    for l in langs:
        save_dir = os.path.join("coco_text_emb", l)
        save_dir_org_text = os.path.join("coco_text_emb", "original_text_feats", l)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir_org_text, exist_ok=True)

        save_p = os.path.join(save_dir, f"{idx}.npy")
        save_p_org_text = os.path.join(save_dir_org_text, f"{idx}.npy")

        
        if l == 'English':
            if not os.path.exists(save_p):
                feat = get_clip_emb(clip_model, clip_processor, row[l])
        else:
            if not os.path.exists(save_p) or not os.path.exists(save_p_org_text):
                feat, org_feat = get_labse_emb(model, labse, row[l])

                torch.save(feat, save_p)
                torch.save(org_feat, save_p_org_text)


##########################################################################################################################


exit()
seed, seed1, seed2 = 0,0,0

df = None
if args.nhead == -1:
    df = pd.read_csv(args.csv_file)
else:
    df = pd.read_csv(args.csv_file).head(args.nhead)
prompts = df[args.text_col].tolist()
image_names = df['file_name'].tolist()

with open("/raid/nlp/ishapandey/tmp/t-code2/lang_col_map.json", 'r') as f:
    lang_col_map = json.load(f)
print(lang_col_map)
args.image_save_dir = os.path.join(args.image_save_dir, args.method, lang_col_map.get(args.text_col, args.text_col))
os.makedirs(args.image_save_dir, exist_ok=True)
print(f"Saved dir is: {args.image_save_dir}")

batch_size = args.batch_size
org_img_name_len = len(image_names)
org_prompt_len = len(prompts)

image_names = [
        image_names[i: i+batch_size] 
        for i in range(0, len(image_names), batch_size) 
]
prompts = [
        prompts[i: i+batch_size] 
        for i in range(0, len(prompts), batch_size) 
]

new_img_len = sum([len(x) for x in image_names])
new_prompt_len = sum([len(x) for x in prompts])

assert org_img_name_len == new_img_len, "save path len not matching, after batching"
assert org_prompt_len == new_prompt_len, "prompt len not matching, after batching"


with torch.no_grad():
    if args.method not in ['clip_only', 'baseline', 'T5_only']:
        pipe._get_clip_prompt_embeds = partial(_get_clip_prompt_embeds, pipe, custom_model=model, custom_labse=labse)
        print("Overwriting")

    for idx, (save_p, p) in tqdm(enumerate(zip(image_names, prompts)), total=len(image_names), desc="Generating images"):
        
        text1 = ""
        text2 = ""

        if args.method == "baseline":
            text1 = p
            text2 = p

        elif args.method == "clip_only":
            text1 = p
            text2 = [args.prompt_2]*len(p)

        elif args.method == "T5_only":
            text1 = [args.prompt_2]*len(p)
            text2 = p
        
        else:
            # like clip only setting
            # but clip feature extractor is changed
            # to aligned models
            text1 = p
            text2 = [args.prompt_2]*len(p)

        print("Text1:", text1)
        print("Text2:", text2)


        save_paths = [
                os.path.join(args.image_save_dir, x)
                for x in save_p
        ]
        
        if all(os.path.exists(p) for p in save_paths):
            print(f"Already exists: {save_p}")
            continue

        image = pipe(
            text1,
            prompt_2= text2,
            height=512,
            width=512,
            guidance_scale=3.5,
            num_inference_steps=10,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(seed)
        )
        
        #image.images[0].save(save_p)

        for path, img in zip(save_paths, image.images):
            img.save(path)
            print(f"Image saved at: {path}")


