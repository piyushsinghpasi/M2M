import torch
import os
import pandas as pd
import pickle
from PIL import Image
import requests

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, eng_file_paths, non_eng_file_paths,
                labse_model=None, clip_model=None, clip_processor=None, 
                clip_text_to_id_path=None, labse_text_to_id_path=None,
                feature_save_dir=None, image_model = 'clip', text_model = 'labse',
        ):
        # NOTE: for parallel data
        # make sure eng_file_paths and non_eng_file_paths
        # are aligned

        self.eng_file_paths = eng_file_paths
        self.non_eng_file_paths = non_eng_file_paths

        # TODO:
        # currently assumes len(eng_data) == len(non_eng_data)
        # one can have diff size for both
        # but then batching becomes tricky to do
        # think on it later
        self.eng_data = []
        self.non_eng_data = []

        self.labse_model = labse_model
        self.clip_model = clip_model
        self.clip_processor = clip_processor

        self.device = clip_model.device

        
        self.image_model = image_model
        self.text_model = text_model

        self.gather_data()

        self.feature_save_dir = feature_save_dir
        self.clip_text_to_id_path = clip_text_to_id_path
        self.labse_text_to_id_path = labse_text_to_id_path

        self.clip_text_to_id = self.create_text_to_id(
                self.clip_text_to_id_path, self.eng_data
        )
        self.labse_text_to_id = self.create_text_to_id(
                self.labse_text_to_id_path, self.eng_data+self.non_eng_data
        )


        with open(self.clip_text_to_id_path, 'wb') as f:
            pickle.dump(self.clip_text_to_id, f)
        
        with open(self.labse_text_to_id_path, 'wb') as f:
            pickle.dump(self.labse_text_to_id, f)

    def gather_data(self):
        # extract data from text files
        # from base_paths and save in data

        # eng data
        for file_path in self.eng_file_paths:
            with open(file_path, 'r') as f:
                eng_lines = f.readlines()

            eng_lines = [l.strip() for l in eng_lines]
            self.eng_data.extend(eng_lines)

        # non eng data
        for file_path in self.non_eng_file_paths:
            with open(file_path, 'r') as f:
                non_eng_lines = f.readlines()

            non_eng_lines = [l.strip() for l in non_eng_lines]
            self.non_eng_data.extend(non_eng_lines)
    
    def create_text_to_id(self, text_to_id_path, text_data):
        text_to_id = dict()
        if os.path.exists(text_to_id_path):
            print(text_to_id_path)
            with open(text_to_id_path, 'rb') as f:
                text_to_id = pickle.load(f)
        
        # add all new texts
        for text in text_data:
            if text not in text_to_id:
                text_to_id[text] = len(text_to_id)

        return text_to_id

    def __len__(self):
        return len(self.eng_data)

    def __getitem__(self, idx):
        eng_text = self.eng_data[idx]
        non_eng_text = self.non_eng_data[idx] if self.non_eng_data else None

        eng_clip_emb = None
        eng_labse_emb = None
        non_eng_labse_emb = None

        if self.clip_model is not None:
            eng_text_clip_id = self.clip_text_to_id[eng_text]
            eng_text_clip_emb_path = os.path.join(
                    self.feature_save_dir, "train", self.image_model, f"{eng_text_clip_id}.pth"
                ) 
            os.makedirs(os.path.dirname(eng_text_clip_emb_path), exist_ok=True)
            if os.path.exists(eng_text_clip_emb_path):
                eng_clip_emb = torch.load(eng_text_clip_emb_path)
            else:
                inputs = self.clip_processor(
                    text=[eng_text], 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )
                
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                      inputs[k] = v.to(self.device)

                eng_clip_emb = self.clip_model.get_text_features(**inputs).detach().cpu()
                torch.save(eng_clip_emb, eng_text_clip_emb_path)

        if self.labse_model is not None:
            
            eng_text_labse_id = self.labse_text_to_id[eng_text]
            eng_text_labse_emb_path = os.path.join(
                    self.feature_save_dir, "train", self.text_model, 
                    f"{eng_text_labse_id}.pth"
            )
            os.makedirs(os.path.dirname(eng_text_labse_emb_path), exist_ok=True)
            if os.path.exists(eng_text_labse_emb_path):
                eng_labse_emb = torch.load(eng_text_labse_emb_path)
            else:
                if self.text_model in ["jinaTextv3"]:
                    eng_labse_emb = torch.from_numpy(self.labse_model.encode([eng_text], task="text-matching"))
                else:
                    eng_labse_emb = torch.from_numpy(self.labse_model.encode([eng_text]))
                torch.save(eng_labse_emb, eng_text_labse_emb_path)
            
           
            if non_eng_text is not None:
                non_eng_text_labse_id = self.labse_text_to_id[non_eng_text]
                non_eng_text_labse_emb_path = os.path.join(
                        self.feature_save_dir, "train", self.text_model, 
                        f"{non_eng_text_labse_id}.pth"
                )
                if os.path.exists(non_eng_text_labse_emb_path):
                    non_eng_labse_emb = torch.load(non_eng_text_labse_emb_path)
                else:
                    if self.text_model in ["jinaTextv3"]:
                        non_eng_labse_emb = torch.from_numpy(self.labse_model.encode([non_eng_text], task="text-matching"))
                    else:
                        non_eng_labse_emb = torch.from_numpy(self.labse_model.encode([non_eng_text]))
                    torch.save(non_eng_labse_emb, non_eng_text_labse_emb_path)

        return {
            'eng_clip_emb': eng_clip_emb,
            'eng_labse_emb': eng_labse_emb,
            'non_eng_labse_emb': non_eng_labse_emb if non_eng_labse_emb is not None else [],
        }


"""## Evaluation Dataset"""

# Initialize CLIP model and processor
class EmbeddingPrecomputator:
    def __init__(self, img_text_df, clip_model, clip_processor, labse_model, device,
                 image_dir, feature_save_dir, lang_cols, img_col, text_to_id_path = None,
                image_model = "clip", text_model = "labse"
        ):
        self.processor = clip_processor
        self.model = clip_model
        self.device = device
        self.labse_model = labse_model
        self.img_text_df = img_text_df
        self.image_dir = image_dir
        self.feature_save_dir = feature_save_dir
        self.lang_cols = lang_cols
        self.img_col = img_col
        self.text_to_id_path = text_to_id_path
        self.image_model = image_model
        self.text_model = text_model
        self.max_length_64 = 0

        # IMP: dont remove 'images' prefix
        # TEXT feats from image-text model only use image_model varible
        # to store features
        self.image_feat_dir = "images"
        if self.image_model != "clip":
            self.image_feat_dir = f"images_{self.image_model}"

        if os.path.exists(text_to_id_path):
            with open(text_to_id_path, 'rb') as f:
                self.text_to_id = pickle.load(f)
        else:
            self.text_to_id = dict()

    @torch.no_grad()
    def get_image_embedding(self, image_name):
        image_name = image_name.split('/')[-1]
        feat_save_path = os.path.join(self.feature_save_dir, self.image_feat_dir, image_name.split('.')[0] + '.pt')
        os.makedirs(os.path.dirname(feat_save_path), exist_ok=True)

        if os.path.exists(feat_save_path):
            image_embedding = torch.load(feat_save_path)
        else:
            image = Image.open(os.path.join(self.image_dir, image_name))
            inputs = self.processor(images=[image], return_tensors="pt", padding=True).to(self.device)
            image_embedding = self.model.get_image_features(**inputs)
            image_embedding = image_embedding.detach().cpu()
            torch.save(image_embedding, feat_save_path)

        return image_embedding


    @torch.no_grad()
    def get_text_embedding(self, text, lang, use_clip=False):
        if text not in self.text_to_id:
            self.text_to_id[text] = len(self.text_to_id)

        text_id = self.text_to_id[text]
        if use_clip:
            feat_save_path = os.path.join(self.feature_save_dir, lang, self.image_model, f'{text_id:05d}.pt')
        else:
            feat_save_path = os.path.join(self.feature_save_dir, lang, self.text_model, f'{text_id:05d}.pt')
        
        os.makedirs(os.path.dirname(feat_save_path), exist_ok=True)

        if False and os.path.exists(feat_save_path):
            text_embedding = torch.load(feat_save_path)
        elif use_clip:
            # IMPORTANT: for siglip models use max_length=64
            if "siglip" in self.image_model.lower():
                inputs = self.processor(text=[text.lower()], 
                    max_length=64, padding="max_length", 
                    return_tensors="pt", 
                    truncation=True
                ).to(self.device)
            else:
                # original
                inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            
            text_embedding = self.model.get_text_features(**inputs)
            text_embedding = text_embedding.detach().cpu()
            torch.save(text_embedding, feat_save_path)
        else:
            if self.text_model in ["jinaTextv3"]:
                text_embedding = self.labse_model.encode([text], task="text-matching")
            else:
                text_embedding = self.labse_model.encode([text])
            text_embedding = torch.from_numpy(text_embedding)
            torch.save(text_embedding, feat_save_path)

        return text_embedding

