import os
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm

# Load your data
# embeddings = np.load("your_embeddings.npy")  # shape (n_sentences × n_langs, dim)
# languages = [...]  # length n_sentences × n_langs
# sentence_ids = [...]  # same length, integer ID per sentence


df = pd.read_csv("eval_data/coco2014/MSCOCO-30K-10-lang.csv") 
df = df.rename(columns={"caption": "English"})

#df = df.sample(n=10000, random_state=2025).reset_index(drop=True)
print(df)
langs = ['English', 'French', 'Greek', 'Hebrew', 'Indonesian',
               'Korean', 'Persian', 'Russian', 'Spanish', 'Hindi']


# diverse sampling

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import numpy as np
import random

np.random.seed(2025)

# Assume you have:
# english_embeddings: NumPy array of shape (N_english_sentences, D)
# english_sentences: list of N_english_sentences
english_embeddings = []
english_sentences = []
for idx, row in tqdm(df.iterrows(), desc="extracting english"):
    print(idx)
    save_dir = os.path.join("coco_text_emb", "English")
    save_p = os.path.join(save_dir, f"{idx}.npy")

    emb = torch.load(save_p)
    english_embeddings.append(emb)

    english_sentences.append(row['English'])


english_embeddings = np.concatenate(english_embeddings, axis=0)
print("English embs", english_embeddings.shape)

# Step 1: Cluster to get semantically similar groups
num_clusters = 100
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(english_embeddings)

# Step 2: Sample ~15 similar sentences from random clusters
similar_sentences = []
sample_cluster_size = 15
diff_cluster_size = 55

# Step 3: Compute pairwise distances
dist_matrix = pairwise_distances(english_embeddings)
avg_dist = dist_matrix.mean(axis=1)

# Step 4: Pick sentences with highest average distance
diverse_idxs = np.argsort(avg_dist)[-diff_cluster_size:]

for cluster_id in np.random.choice(range(num_clusters), size=sample_cluster_size, replace=False):
    idxs = np.where(cluster_labels == cluster_id)[0]
    idxs = [x for x in idxs if x not in diverse_idxs]
    if len(idxs) >= 3:
        sampled = np.random.choice(idxs, size=3, replace=False)
        similar_sentences.extend(sampled.tolist())

# Combine the indices
final_indices = sorted(set(similar_sentences + diverse_idxs.tolist()))

# Show selected sentences
selected_sentences = [english_sentences[i] for i in final_indices]

for i, (sent, idx) in enumerate(zip(selected_sentences, final_indices)):
    print(f"{idx}- {i+1:02d}. {sent}")
    print(df.iloc[idx])



embeddings = []
languages = []
sentence_ids = []

print("df 2", df.shape)
for idx, row in tqdm(df.iterrows(), desc="extracting"):
    if idx not in final_indices : continue
    for l in langs:
        save_dir = os.path.join("coco_text_emb", l)
        save_p = os.path.join(save_dir, f"{idx}.npy")

        emb = torch.load(save_p)
        embeddings.append(emb)

        languages.append(l)
        sentence_ids.append(idx)

embeddings = np.concatenate(embeddings, axis=0)
print("embs", embeddings.shape)

# Run t-SNE
pps = [30, 32, 50, 80, 100]

for pp in pps:
    print("Running tsne", pp)
    tsne = TSNE(n_components=2, random_state=42, perplexity=pp, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(embeddings)

    # DataFrame for easy plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'language': languages,
        'sentence_id': sentence_ids
    })


    # Optional jitter
    df['x'] = df['x'] + np.random.normal(0, 0.05, size=len(df))
    df['y'] = df['y'] + np.random.normal(0, 0.05, size=len(df))

    # -----------------------------
    # Color by sentence ID
    num_sentences = df['sentence_id'].nunique()
    cmap = plt.get_cmap("gist_rainbow", num_sentences)
    sentence_color_map = {
        sid: cmap(i) for i, sid in enumerate(sorted(df['sentence_id'].unique()))
    }

    # Marker shape by language
    unique_langs = sorted(df['language'].unique())
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', '*']
    language_marker_map = {
        lang: markers[i % len(markers)] for i, lang in enumerate(unique_langs)
    }

    # -----------------------------
    # Plot setup
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='datalim')

    # Plot all points
    for lang in unique_langs:
        lang_df = df[df['language'] == lang]
        plt.scatter(
            lang_df['x'],
            lang_df['y'],
            c=[sentence_color_map[sid] for sid in lang_df['sentence_id']],
            marker=language_marker_map[lang],
            label=lang,
            alpha=0.5,
            s=50,
            edgecolors='black',
            linewidths=0.2
        )

    # -----------------------------
    # Draw circles around clusters (random subset)
    np.random.seed(42)
    #highlight_sids = np.random.choice(df['sentence_id'].unique(), size=50, replace=False)
    percentile = 75
    for sid in df['sentence_id'].unique():
        sid_df = df[df['sentence_id'] == sid]
        points = sid_df[['x', 'y']].values
        if len(points) >= 2:
            centroid = points.mean(axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)
            #radius = distances.max()
            radius = np.percentile(distances, percentile)
            #print("r", radius)
            #radius = max(2, radius)
            
            radius = 2
            circle = patches.Circle(
                centroid,
                radius,
                edgecolor='black',#sentence_color_map[sid],
                facecolor='none',#sentence_color_map[sid],
                linewidth=1.0,
                alpha=0.6
            )
            ax.add_patch(circle)
    
    # -----------------------------
    # Custom legend: shapes only
    legend_handles = [
        Line2D(
            [0], [0],
            marker=language_marker_map[lang],
            color='white',
            label=lang,
            linestyle='None',
            markerfacecolor='gray',
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=0.5
        ) for lang in unique_langs
    ]

    plt.legend( title="Language", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("t-SNE of Parallel Multilingual Embeddings\nColor = Sentence ID, Marker = Language", fontsize=16)
    plt.xlabel("t-SNE Dim 1 (scaled)")
    plt.ylabel("t-SNE Dim 2 (scaled)")
    plt.tight_layout()
    save_p = f"images_tsne/tsne_multilingual_plot_{pp}.png"
    plt.savefig(save_p, bbox_inches='tight', transparent=False)

    print(f"Saved at {save_p}")
    #plt.show()
