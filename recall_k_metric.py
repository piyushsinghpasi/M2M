import numpy as np
from collections import defaultdict
import torch
from sentence_transformers import util


def calculate_recall_at_k(similarity_matrix, ground_truth, k):
    """
    Calculate recall@k.

    Args:
        similarity_matrix (np.ndarray): Similarity matrix (rows: queries, cols: candidates).
        ground_truth (list of sets): Ground truth indices for each query.
        k (int): Value of K for recall@K.

    Returns:
        float: Recall@K value.
    """
    num_queries = len(ground_truth)
    recall_sum = 0

    for query_idx, true_indices in enumerate(ground_truth):
        top_k_indices = np.argsort(-similarity_matrix[query_idx])[:k]

        if isinstance(top_k_indices, torch.Tensor) or isinstance(top_k_indices, np.ndarray):
            top_k_indices = top_k_indices.tolist()
        
        if any(idx in true_indices for idx in top_k_indices):
            recall_sum += 1

    return 100*recall_sum / num_queries

def retrieval_evaluation(
        image_to_captions, 
        text_to_images, 
        get_image_feature,
        get_text_feature,
        k_values=[1, 5, 10],
        device='cuda',
        img_ext = "", 
        cos_util=False, 
        precomputed_features = None
    ):
    """
    Evaluate text-to-image and image-to-text retrieval.

    Args:
        image_to_captions (dict): Mapping of image path to list of captions.
        text_to_images (dict): Mapping of caption to list of image paths.
        get_image_feature: a function which takes image path, return 1xD feature
        get_text_feature: a function which takes text, return 1xD feature
        k_values (list): List of K values for recall@K.

    Returns:
        dict: Results containing recall@K for both text-to-image and image-to-text.
    """
    if precomputed_features is None:
        images = list(image_to_captions.keys())
        
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
        # I x D
        image_features = torch.cat([get_image_feature(img+img_ext) for img in images], dim=0)
        
        print("text", text_features.shape)
        print("image", image_features.shape)

    else:
        print("using precomputed features")
        images = precomputed_features["images"]
        captions = precomputed_features["captions"]
        text_features = precomputed_features["text_features"]
        image_features = precomputed_features["image_features"]
    
        print("text", text_features.shape)
        print("image", image_features.shape)

    # Compute similarity matrices
    if cos_util:
        text_features = text_features.detach().cpu().numpy()
        image_features = image_features.detach().cpu().numpy()

        text_to_image_similarity = util.cos_sim(text_features, image_features)
        image_to_text_similarity = text_to_image_similarity.T
    else:
        text_features = torch.nn.functional.normalize(text_features, dim=-1).to(device)
        image_features = torch.nn.functional.normalize(image_features, dim=-1).to(device)
        
        text_to_image_similarity = text_features @ image_features.permute(1, 0)
        image_to_text_similarity = text_to_image_similarity.permute(1, 0 )

        text_to_image_similarity = text_to_image_similarity.detach().cpu().numpy()
        image_to_text_similarity = image_to_text_similarity.detach().cpu().numpy()

    # Prepare ground truth for evaluation
    text_to_image_ground_truth = [
        {images.index(img) for img in text_to_images[caption] if img in images} for caption in captions
    ]
    image_to_text_ground_truth = [
        {captions.index(caption) for caption in image_to_captions[img] if caption in captions} for img in images
    ]

    # Evaluate recall@K
    results = {
        "text_to_image": {},
        "image_to_text": {}
    }

    for k in k_values:
        results["text_to_image"][f"recall@{k}"] = calculate_recall_at_k(
            text_to_image_similarity, text_to_image_ground_truth, k
        )
        results["image_to_text"][f"recall@{k}"] = calculate_recall_at_k(
            image_to_text_similarity, image_to_text_ground_truth, k
        )

    return results


