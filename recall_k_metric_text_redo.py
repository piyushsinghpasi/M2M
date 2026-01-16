import numpy as np
from collections import defaultdict
import torch
from sentence_transformers import util


def calculate_recall_at_k(similarity_matrix, ground_truth, k, text_text_sim=None, captions=None, text_id_to_caption_map=None, images=None, image_to_captions=None):
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
    tt_sum = 0
    for query_idx, true_indices in enumerate(ground_truth):
        top_k_indices = np.argsort(-similarity_matrix[query_idx])[:k+1]

        if isinstance(top_k_indices, torch.Tensor) or isinstance(top_k_indices, np.ndarray):
            top_k_indices = top_k_indices.tolist()

        print("Pred Befor", top_k_indices)
        top_k_indices = [
                x for x in top_k_indices 
                #if x != query_idx
            ][:k]
        
        if any(idx in true_indices for idx in top_k_indices):
            recall_sum += 1

        print("Query idx", query_idx)
        print("GT Index", true_indices)
        print("Pred Index", top_k_indices, similarity_matrix.shape)
        print('-'*100)
        
        # taking top 
        if captions is not None:
            print(f"{k}-GT text", captions[query_idx], text_id_to_caption_map.get(captions[query_idx]))
            for ogt in true_indices:
                print(f"O-GT text", captions[ogt], text_id_to_caption_map.get(captions[ogt]))

            tt_sim = 0
            tot_len = 0
            tt_sims = []
            for pred_idx in top_k_indices:
                tmp_tt_sim = []
                c_= image_to_captions[images[pred_idx]]
                #c_ = [captions[pred_idx]]
                for texts in c_:
                    print("Pred text", texts, text_id_to_caption_map.get(texts, texts))
                    #print("Pred text", texts)
                    tid = captions.index(texts)
                    tmp_tt_sim.append(text_text_sim[query_idx][tid].item())
                    tot_len += 1

                    print("T-T sim",  text_text_sim[query_idx][tid].item())
                
                tt_sims.append(tmp_tt_sim)
                print("Sum:", sum(tt_sims[-1]),"Len",  len(tt_sims[-1]),"Avg", np.mean(tt_sims[-1]))
                print('-'*100)

            print("tt_sims", tt_sims)
            tt_sim_avg = np.mean([np.mean(x).item() for x in tt_sims])
            print("TT sim avg", tt_sim_avg)
            tt_sum += tt_sim_avg
            print("="*100)


    return 100*tt_sum / num_queries

def retrieval_evaluation(
        image_to_captions, 
        text_to_images, 
        get_image_feature,
        get_text_feature,
        k_values=[1, 5, 10],
        device='cuda',
        img_ext = "", 
        cos_util=False, 
        precomputed_features = None,
        get_text_base = None,
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
        text_features_base = torch.cat([
            get_text_base(text_id_to_caption_map.get(caption, caption)) 
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
        
        text_to_text_similarity = util.cos_sim(text_features, text_features)
        img_to_img_similarity = util.cos_sim(image_features, image_features)
    else:
        text_features = torch.nn.functional.normalize(text_features, dim=-1).to(device)
        image_features = torch.nn.functional.normalize(image_features, dim=-1).to(device)
        
        text_to_image_similarity = text_features @ image_features.permute(1, 0)
        image_to_text_similarity = text_to_image_similarity.permute(1, 0 )

        text_to_image_similarity = text_to_image_similarity.detach().cpu().numpy()
        image_to_text_similarity = image_to_text_similarity.detach().cpu().numpy()
       
        text_features_base =  torch.nn.functional.normalize(text_features_base, dim=-1).to(device)
        #text_features_base = text_features; input("Overwriten")
    
        text_to_text_similarity = text_features_base @ text_features_base.permute(1,0)
        text_to_text_similarity = text_to_text_similarity.detach().cpu().numpy()
        
        img_to_img_similarity = image_features @ image_features.permute(1,0)

    # Prepare ground truth for evaluation
    text_to_image_ground_truth = [
        {images.index(img) for img in text_to_images[caption] if img in images} for caption in captions
    ]
    
    text_idx_of_img_for_Qtext = []
    for Qidx, img_idxs in enumerate(text_to_image_ground_truth):
        text_idx_of_img_for_Qtext_ = []
        for img_idx in img_idxs:
            texts_of_img_idx = image_to_captions[images[img_idx]]
            text_idxs_of_img_idx = [captions.index(t) for t in texts_of_img_idx]
            text_idx_of_img_for_Qtext_.extend(text_idxs_of_img_idx)
        
        print(Qidx, text_idx_of_img_for_Qtext_)
        text_idx_of_img_for_Qtext_ = set(text_idx_of_img_for_Qtext_)

        # remove current Qidx, only keep similar text, not identical
        text_idx_of_img_for_Qtext_ = {x for x in text_idx_of_img_for_Qtext_ if x!=Qidx}
        text_idx_of_img_for_Qtext.append(set(text_idx_of_img_for_Qtext_))

    image_to_text_ground_truth = [
        {captions.index(caption) for caption in image_to_captions[img] if caption in captions} for img in images
    ]

    # Evaluate recall@K
    results = {
        "text_to_image": {},
        "image_to_text": {},
        "text_to_text": {},
    }
    
    #k_values = [1]
    for k in k_values:
        
        results["text_to_image"][f"recall@{k}"] = calculate_recall_at_k(
            text_to_image_similarity, text_to_image_ground_truth, k, text_to_text_similarity, captions, text_id_to_caption_map, images, image_to_captions,
        )
        
        '''
        results["image_to_text"][f"recall@{k}"] = calculate_recall_at_k(
            image_to_text_similarity, image_to_text_ground_truth, k, img_to_img_similarity, images, dict(), captions, text_to_images,
        )
        '''
        '''
        results["text_to_text"][f"recall@{k}"] = calculate_recall_at_k(
            text_to_text_similarity, text_idx_of_img_for_Qtext, k,
            text_to_text_similarity, captions, text_id_to_caption_map, images, image_to_captions,
        )
        '''

    print(results)
    return results


