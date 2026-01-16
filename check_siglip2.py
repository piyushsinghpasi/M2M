from transformers import pipeline

# load pipeline
ckpt = "google/siglip2-giant-opt-patch16-384"

# load image and candidate labels
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
candidate_labels = ["2 cats", "a plane", "a remote"]
candidate_labels = ["2 Katzen", "ein Flugzeug", "eine Fernbedienung"]

# run inference


import torch
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image

# load the model and processor
DEVICE="cuda"
ckpt = "google/siglip2-giant-opt-patch16-384"
model = AutoModel.from_pretrained(ckpt).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(ckpt)

# load the image
#image = load_image("https://huggingface.co/datasets/merve/coco/resolve/main/val2017/000000000285.jpg")
image = load_image(url)
inputs = processor(images=[image], return_tensors="pt").to(model.device)

# run infernece
with torch.no_grad():
    image_embeddings = model.get_image_features(**inputs)    

    #text_inps = processor(text=candidate_labels, padding=True, truncation=True, return_tensors="pt").to(model.device)
    text_inps = processor(text=candidate_labels, max_length=64, padding="max_length", return_tensors="pt").to(model.device)
    text_embs = model.get_text_features(**text_inps)

    text_embs = torch.nn.functional.normalize(text_embs, p=2, dim=-1)
    image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)

    print(image_embeddings.shape)
    print(text_embs.shape)

    print(image_embeddings @ text_embs.T)
