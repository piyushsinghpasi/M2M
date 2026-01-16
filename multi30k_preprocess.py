import os
import pandas as pd
import pickle

base_path = "/home/piyush/labse-clip/eval_data/multi30k-dataset"
base_image_dir = os.path.join(base_path, "flickr30k", "Images")
# Define the file paths
file_paths = ['image_splits/test_2016_flickr.txt', 
        'test_2016_flickr.cs', 'test_2016_flickr.de', 
        'test_2016_flickr.en', 'test_2016_flickr.fr'
]

# Read the files and store the content in a dictionary

data = {}
for f in file_paths:
    with open(os.path.join(base_path, f), 'r') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]

    col_name = f.split('.')[-1]
    if "image" in f:
        col_name = "image_name"
        for image_name in lines:
            if not os.path.join(base_image_dir, image_name):
                print(image_name)
    data[col_name] = lines


# Create a DataFrame from the dictionary
img_text_df = pd.DataFrame(data)  # Using index [0] to create a single row

# Display the DataFrame
print(img_text_df)
print(img_text_df.columns)

lang_cols = ['en', 'de', 'cs', 'fr']
text_to_img_map = {l: dict() for l in lang_cols}
img_to_text_map = {l: dict() for l in lang_cols}

imgs = img_text_df["image_name"].tolist()

for l in lang_cols:
    print(f"Mapping {l}")
    texts = img_text_df[l].tolist()
    
    for k,v in zip(imgs, texts):
        k = os.path.join(base_image_dir, k)
        
        if v in text_to_img_map[l]:
            print("duplicate", text_to_img_map[l][v], v, k)
            # this makes texts unique for 1-1 retrieval
            # this is removed before passing to model
            v += SPACER
            #raise ValueError("duplicate text")
        text_to_img_map[l][v] = [k]
        
        if k in img_to_text_map[l]:
            raise ValueError("duplicate images")    
        img_to_text_map[l][k] = [v]


img_to_text_map_path = "eval_data/multi30k-dataset/image_to_text_mapping.pkl"
with open(img_to_text_map_path, "wb") as f:
    pickle.dump(img_to_text_map, f)

text_to_img_map_path = "eval_data/multi30k-dataset/text_to_image_mapping.pkl"
with open(text_to_img_map_path, "wb") as f:
    pickle.dump(text_to_img_map, f)

