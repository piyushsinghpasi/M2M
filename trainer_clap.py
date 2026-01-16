import torch
import os
import pickle
import numpy as np
import wandb
from tqdm import tqdm
import json
from functools import partial
#from recall_k_metric_text_redo import retrieval_evaluation
from recall_k_metric import retrieval_evaluation

"""# **TRAINER**"""

class Trainer:
    def __init__(self, model, train_loader, eval_dataset, optimizer, scheduler, device, k_values, args):
        self.model = model
        self.train_loader = train_loader
        self.eval_dataset = eval_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.k_values = k_values
        self.args = args
        self.best_metric = -999
        self.best_epoch = 0

    def train(self, start_epoch, end_epoch):

        for epoch in range(start_epoch, end_epoch+1):
            self.model.train()
            epoch_loss, epoch_loss_mse, epoch_loss_kl, epoch_loss_kl_clip = 0., 0., 0., 0.
            epoch_loss_eng_cosine = 0.

            for idx, batch in tqdm(enumerate(self.train_loader), desc=f"Epoch {epoch}", total=len(self.train_loader)):

                # Prepare the inputs
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)


                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch)

                # Compute the loss (example: using MSE loss)
                # Replace with the actual loss function from your model
                if self.args.loss_method == "mse":
                    loss_mse = self.model.mse_loss(outputs)
                    loss = loss_mse
                    loss_kl = torch.zeros_like(loss_mse)
                    loss_kl_clip = torch.zeros_like(loss_mse)
                elif self.args.loss_method == "l1":
                    loss_mse = self.model.l1_loss(outputs)
                    loss = loss_mse
                    loss_kl = torch.zeros_like(loss_mse)
                    loss_kl_clip = torch.zeros_like(loss_mse)
                elif self.args.loss_method == "mse_kl":
                    loss_mse = self.model.mse_loss(outputs)
                    loss_kl = self.model.kl_loss(outputs, use_mse=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl*loss_kl
                    loss_kl_clip = torch.zeros_like(loss_mse)
                elif self.args.loss_method == "l1_kl":
                    loss_mse = self.model.l1_loss(outputs)
                    loss_kl = self.model.kl_loss(outputs, use_mse=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl*loss_kl
                    loss_kl_clip = torch.zeros_like(loss_mse)
                elif self.args.loss_method == "mse_newkl":
                    loss_mse = self.model.mse_loss(outputs)
                    loss_kl = self.model.new_kl_loss(outputs, use_mse=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl*loss_kl
                    loss_kl_clip = torch.zeros_like(loss_mse)
                elif self.args.loss_method == "l1_newkl":
                    loss_mse = self.model.l1_loss(outputs)
                    loss_kl = self.model.new_kl_loss(outputs, use_mse=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl*loss_kl
                    loss_kl_clip = torch.zeros_like(loss_mse)
                elif self.args.loss_method == "l1_kll1":
                    loss_mse = self.model.l1_loss(outputs)
                    loss_kl = self.model.kl_loss(outputs, use_l1=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl*loss_kl
                    loss_kl_clip = torch.zeros_like(loss_mse)
                elif self.args.loss_method == "mse_klclip":
                    loss_mse = self.model.mse_loss(outputs)
                    loss_kl_clip = self.model.kl_clip_loss(outputs, use_mse=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl_clip*loss_kl_clip
                    loss_kl = torch.zeros_like(loss_mse)
                elif self.args.loss_method == "l1_klclip":
                    loss_mse = self.model.l1_loss(outputs)
                    loss_kl_clip = self.model.kl_clip_loss(outputs, use_mse=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl_clip*loss_kl_clip
                    loss_kl = torch.zeros_like(loss_mse)
                elif self.args.loss_method == "l1_klclipl1":
                    loss_mse = self.model.l1_loss(outputs)
                    loss_kl_clip = self.model.kl_clip_loss(outputs, use_l1=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl_clip*loss_kl_clip
                    loss_kl = torch.zeros_like(loss_mse)
                elif self.args.loss_method == "l1_kl_klclip":
                    loss_mse = self.model.l1_loss(outputs)
                    loss_kl = self.model.kl_loss(outputs, use_mse=True)
                    loss_kl_clip = self.model.kl_clip_loss(outputs, use_mse=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl*loss_kl + self.args.c_kl_clip*loss_kl_clip
                elif self.args.loss_method == "mse_kl_klclip":
                    loss_mse = self.model.mse_loss(outputs)
                    loss_kl = self.model.kl_loss(outputs, use_mse=True)
                    loss_kl_clip = self.model.kl_clip_loss(outputs, use_mse=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl*loss_kl + self.args.c_kl_clip*loss_kl_clip
                elif self.args.loss_method == "l1_kll1_klclipl1":
                    loss_mse = self.model.l1_loss(outputs)
                    loss_kl = self.model.kl_loss(outputs, use_l1=True)
                    loss_kl_clip = self.model.kl_clip_loss(outputs, use_l1=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl*loss_kl + self.args.c_kl_clip*loss_kl_clip
                elif self.args.loss_method == "mse_kll1_klclipl1":
                    loss_mse = self.model.mse_loss(outputs)
                    loss_kl = self.model.kl_loss(outputs, use_l1=True)
                    loss_kl_clip = self.model.kl_clip_loss(outputs, use_l1=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl*loss_kl + self.args.c_kl_clip*loss_kl_clip
                elif self.args.loss_method == "l1_newkl_klclip":
                    loss_mse = self.model.l1_loss(outputs)
                    loss_kl = self.model.new_kl_loss(outputs, use_mse=True)
                    loss_kl_clip = self.model.kl_clip_loss(outputs, use_mse=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl*loss_kl + self.args.c_kl_clip*loss_kl_clip
                elif self.args.loss_method == "mse_newkl_klclip":
                    loss_mse = self.model.mse_loss(outputs)
                    loss_kl = self.model.new_kl_loss(outputs, use_mse=True)
                    loss_kl_clip = self.model.kl_clip_loss(outputs, use_mse=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_kl*loss_kl + self.args.c_kl_clip*loss_kl_clip
                elif self.args.loss_method == "mse_klclip_eng_cosine":
                    loss_mse = self.model.mse_loss(outputs)
                    loss_eng_cosine = self.model.eng_cosine_loss(outputs)
                    loss_kl_clip = self.model.kl_clip_loss(outputs, use_mse=True)
                    loss = self.args.c_dist*loss_mse + self.args.c_eng_cosine*loss_eng_cosine + self.args.c_kl_clip*loss_kl_clip
                    loss_kl = torch.zeros_like(loss_mse)
                elif self.args.loss_method == "eng_cosine":
                    loss_eng_cosine = self.model.eng_cosine_loss(outputs)
                    loss_mse = torch.zeros_like(loss_mse)
                    loss_kl_clip = torch.zeros_like(loss_mse)
                    loss_kl = torch.zeros_like(loss_mse)
                    
                    loss = loss_eng_cosine 
                
                if 'eng_cosine' not in self.args.loss_method:
                    loss_eng_cosine = torch.zeros_like(loss_mse)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                lr = self.optimizer.param_groups[0]['lr']
                #print(f"Epoch: {epoch}, Batch: {idx+1}, Loss: {loss.item():6f}, LR: {lr}")

                epoch_loss += loss.item()
                epoch_loss_mse += loss_mse.item()
                epoch_loss_kl += loss_kl.item()
                epoch_loss_kl_clip += loss_kl_clip.item()
                epoch_loss_eng_cosine += loss_eng_cosine.item()

                logs = {
                    "train/epoch": epoch,
                    "train/step_loss_mse": loss_mse.item(),
                    "train/step_loss_kl": loss_kl.item(),
                    "train/step_loss_kl_clip": loss_kl_clip.item(),
                    "train/step_loss_eng_cosine": loss_eng_cosine.item(),
                    "train/step_loss": loss.item(),
                    "train/lr": lr,
                }
                wandb.log(logs)

            # avged over all batches
            wandb.log({
                "train/epoch_loss": epoch_loss/len(self.train_loader),
                "train/epoch_loss_mse": epoch_loss_mse/len(self.train_loader),
                "train/epoch_loss_kl": epoch_loss_kl/len(self.train_loader),
                "train/epoch_loss_kl_clip": epoch_loss_kl_clip/len(self.train_loader),
                "train/epoch_loss_eng_cosine": epoch_loss_eng_cosine/len(self.train_loader),
            })


            metrics = self.evaluate(use_clip=False)
            wandb.log(
                {
                    "eval": metrics,
                }
            )

            avg_en_t2i = np.mean([metrics['en']['text_to_image'][f'recall@{k}'] for k in [1,5,10]])
            avg_en_i2t = np.mean([metrics['en']['image_to_text'][f'recall@{k}'] for k in [1,5,10]])

            avg_en = (avg_en_t2i + avg_en_i2t)/2.0
            #if metrics.get("eval/avg_metric", -1) > self.best_metric:
            if avg_en > self.best_metric:
                #self.best_metric = metrics.get("eval/avg_metric")
                self.best_metric = avg_en
                self.best_epoch = epoch
                print(f"Best epoch {self.best_epoch} and Best metric (en avg): {self.best_metric:.2f}")
                save_path = os.path.join(self.args.save_dir, "checkpoint.pth")
                self.save_model(save_path)

            if epoch == 1:
                # save text_to_id at text_to_id_path after 1st evaluation
                # if text_to_id_path doesn't exist
                
                config_path = os.path.join(self.args.save_dir, "config.json")
                with open(config_path, "w") as f:
                    json.dump(vars(self.args), f)

            if self.args.early_stopping:
                if (epoch - self.best_epoch) > self.args.penalty_gap:
                    break

    @torch.no_grad()
    def evaluate(self, use_clip):
        self.model.eval() 
        langs = sorted(self.eval_dataset.img_to_text_map.keys())
        img_ext = ""

        all_metrics = dict()
        for l in langs:
            #if l!= 'eng_Latn': continue
            if use_clip:
                if "en" == l or "eng_Latn" == l:
                    print(f"Evaluating {l}")
                else:
                    print(f"Skipping {l} as baseline is for English lang. only")
                    continue
            else:
                print(f"Evaluating {l}")

            get_text_feature = partial(self.eval_dataset.get_text_embedding, lang=l, use_clip=use_clip)
            wrapper_func = get_text_feature
            
            wrapper_func_org = get_text_feature
            
            if "baseline" not in self.args.mode:
                def get_text_embed(text, get_t_feat, model):
                    t_feat = get_t_feat(text).to(self.device)
                    t_feat = model.inference(t_feat).detach().cpu()
                    return t_feat
                wrapper_func = partial(get_text_embed, get_t_feat=get_text_feature, model=self.model)

            l_metric = retrieval_evaluation(
                self.eval_dataset.img_to_text_map[l], 
                self.eval_dataset.text_to_img_map[l], 
                get_image_feature=self.eval_dataset.get_audio_embedding,
                get_text_feature=wrapper_func,
                img_ext = img_ext,
                #get_text_base = wrapper_func_org,
            )

            all_metrics[l] = l_metric

            print(l_metric)
            print("="*100)

        return all_metrics


    @torch.no_grad()
    def evaluate2(self, return_top_idx=False, use_clip=False, labse_baseline=False, t2t=False, en_lang_col ='XTD10_captions_en',
            t2t_order=None, # which model1 to which model2   
            skip_inference=False,
        ):
        print('Evaluating...')
        self.model.eval()

        all_imgs = []
        all_text = []
        metrics = dict()
        pred_ranks = dict()

        # MAKE a unique audio only df with GT text index
        for idx, row in self.eval_dataset.audio_only_df.iterrows():
            image_url = str(row[self.eval_dataset.img_col])+".wav"
            image_embedding = self.eval_dataset.get_audio_embedding(image_url)

            all_imgs.append(image_embedding)


        # I and T are same for now
        # I x D
        all_imgs = torch.cat(all_imgs, dim=0).to(self.device)
        all_imgs = torch.nn.functional.normalize(all_imgs, dim=-1, p=2)

        t2t_prefix = ""

        for lang in self.eval_dataset.lang_cols:

            print(f'Evaluating for lang: {lang}')
            all_text = []

            #for idx, row in self.eval_dataset.img_text_df.iterrows():
            for idx, row in self.eval_dataset.audio_only_df.iterrows():
                text_idxs = row['GT_index']
                #query_text = row[lang]  # Assuming 'caption' is the text query column
                for tidx in text_idxs:
                    query_text = self.eval_dataset.img_text_df[lang][tidx]
                    #print(query_text)
                    query_embedding = self.eval_dataset.get_text_embedding(query_text, lang, use_clip=use_clip).to(self.device)

                    # skip model inference for
                    # clip and labse baselines
                    if use_clip or labse_baseline or skip_inference:
                        pass
                    else:
                        query_embedding = self.model.inference(query_embedding).detach().cpu()

                    all_text.append(query_embedding)


            # T x D
            all_text = torch.cat(all_text, dim=0).to(self.device)
            all_text = torch.nn.functional.normalize(all_text, dim=-1, p=2)

            i2t = all_imgs @ all_text.permute(1, 0)
            t2i = i2t.permute(1,0)

            print("i2t pred sim shape", i2t.shape)
            print("t2i pred sim shape", t2i.shape)

            max_k = max(self.k_values)
            i2t_ranks = torch.argsort(i2t, descending=True, dim=1)[:, :max_k].detach().cpu()
            t2i_ranks = torch.argsort(t2i, descending=True, dim=1)[:, :max_k].detach().cpu()

            '''
            print("i2t sim")
            print(i2t)
            print("t2i sim")
            print(t2i)
            '''

            if return_top_idx:
                pred_ranks[lang] = {
                    'i2t': i2t_ranks,
                    't2i': t2i_ranks
                }

            # 2d GT matrix: N x 5
            #i2t_gt_rank = torch.Tensor(self.eval_dataset.audio_only_df['GT_index'].tolist()).long()
            i2t_gt_rank = torch.arange(i2t.shape[1]).view(-1, 5)

            # 1d GT matrix
            #t2i_gt_rank = [self.eval_dataset.tidx_2_aidx[tidx] for tidx in range(t2i_ranks.shape[0])]
            t2i_gt_rank = [j for j in range(i2t.shape[0]) for _ in range(5)]
            t2i_gt_rank = torch.Tensor(t2i_gt_rank).long()
            
            '''
            print("i2t gt rank shape",i2t_gt_rank.shape)
            print("t2i gt rank shape", t2i_gt_rank.shape)
            print("i2t pred ranks shape", i2t_ranks.shape)
            print("t2i pred ranks shape", t2i_ranks.shape)
            
            print("i2t pred ranks", i2t_ranks[:,:10])
            print("i2t GT rank", i2t_gt_rank)
            
            print("t2i Pred rank", t2i_ranks[:,:10])
            print("t2i GT rank", t2i_gt_rank)
            '''

            i2t_is_correct_in_top_k = (i2t_gt_rank[:, None, :] == i2t_ranks[:, :, None]).any(dim=2) # Shape: [T, K]
            t2i_is_correct_in_top_k = (t2i_ranks == t2i_gt_rank[:, None])  # Shape: [T, K]

            # Now, compute Recall@K for each K
            i2t_recall = {f'eval/R@{k}/{t2t_prefix}i2t/{lang}': (i2t_is_correct_in_top_k[:, :k].sum(dim=1) > 0).float().mean().item()*100 for k in self.k_values}
            t2i_recall = {f'eval/R@{k}/{t2t_prefix}t2i/{lang}': (t2i_is_correct_in_top_k[:, :k].sum(dim=1) > 0).float().mean().item()*100 for k in self.k_values}

            metrics.update(i2t_recall)
            metrics.update(t2i_recall)

        # avg across all data/langs
        avg_metric_t2i = np.mean([v for k,v in metrics.items() if "t2i/" in k])
        avg_metric_i2t = np.mean([v for k,v in metrics.items() if "i2t/" in k])
        avg_metric = (avg_metric_t2i + avg_metric_i2t)/2.0

        metrics.update({
            f"eval/{t2t_prefix}avg_metric_t2i": avg_metric_t2i,
            f"eval/{t2t_prefix}avg_metric_i2t": avg_metric_i2t,
            f"eval/{t2t_prefix}avg_metric": avg_metric,
        })
        wandb.log(metrics)

        if return_top_idx:
            return metrics, pred_ranks

        return metrics


    @torch.no_grad()
    def save_model(self, save_path):
        state_dict = self.model.state_dict()
        exclude_layers = ['labse', 'clip']

        filtered_state_dict = {k: v for k, v in state_dict.items() if not any(layer in k for layer in exclude_layers)}

        torch.save(filtered_state_dict, save_path)

