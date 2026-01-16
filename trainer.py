import torch
import os
import pickle
import numpy as np
import wandb
from tqdm import tqdm
import json
import pickle

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

        # for below steps No L_str loss/annealling
        self.L_str_warmup = 1000

    def train(self, start_epoch, end_epoch):

        num_steps = 0
        for epoch in range(start_epoch, end_epoch+1):
            self.model.train()
            epoch_loss, epoch_loss_mse, epoch_loss_kl, epoch_loss_kl_clip = 0., 0., 0., 0.
            epoch_loss_eng_cosine = 0.

            for idx, batch in tqdm(enumerate(self.train_loader), desc=f"Epoch {epoch}", total=len(self.train_loader)):
                num_steps += 1
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
                if self.args.loss_method == "mse_non_eng":
                    loss_mse = self.model.mse_loss_non_eng(outputs)
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
                elif self.args.loss_method == "mse_klclip_warmup":
                    loss_mse = self.model.mse_loss(outputs)
                    if num_steps <= self.L_str_warmup:
                        loss_kl_clip = torch.zeros_like(loss_mse)
                    else:
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
                    loss_mse = torch.zeros_like(loss_eng_cosine)
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

            
            if epoch == 1:
                # save text_to_id at text_to_id_path after 1st evaluation
                # if text_to_id_path doesn't exist
                if not os.path.exists(self.eval_dataset.text_to_id_path):
                  with open(self.eval_dataset.text_to_id_path, 'wb') as f:
                      pickle.dump(self.eval_dataset.text_to_id, f)
                
                config_path = os.path.join(self.args.save_dir, "config.json")
                with open(config_path, "w") as f:
                    json.dump(vars(self.args), f)

            if self.eval_dataset.img_text_df is None:
                print(f"No eval for epoch {epoch}")
                # if no eval data then exit after 10 epochs
                if epoch == 10:
                    break
                else:
                    continue

            metrics = self.evaluate()

            avg_en_t2i = np.mean([metrics[f'eval/R@{k}/t2i/XTD10_captions_en'] for k in [1,5,10]])
            avg_en_i2t = np.mean([metrics[f'eval/R@{k}/i2t/XTD10_captions_en'] for k in [1,5,10]])

            #avg_en_t2i = np.mean([metrics[f'eval/R@{k}/t2i/caption_reference_description'] for k in [1,5,10]])
            #avg_en_i2t = np.mean([metrics[f'eval/R@{k}/i2t/caption_reference_description'] for k in [1,5,10]])

            avg_en = (avg_en_t2i + avg_en_i2t)/2.0
            
            if avg_en > self.best_metric:
                #self.best_metric = metrics.get("eval/avg_metric")
                self.best_metric = avg_en
                self.best_epoch = epoch
                print(f"Best epoch {self.best_epoch} and Best metric (en avg): {self.best_metric:.2f}")
                save_path = os.path.join(self.args.save_dir, "checkpoint.pth")
                self.save_model(save_path)


            if self.args.early_stopping:
                if (epoch - self.best_epoch) > self.args.penalty_gap:
                    break


    @torch.no_grad()
    def evaluate(self, return_top_idx=False, use_clip=False, labse_baseline=False, t2t=False, en_lang_col ='XTD10_captions_en',
            t2t_order=None, # which model1 to which model2   
            skip_inference=False, save_feat_dir = None, 
        ):
        print('Evaluating...')
        self.model.eval()

        all_imgs = []
        all_text = []
        metrics = dict()
        pred_ranks = dict()

        for idx, row in tqdm(self.eval_dataset.img_text_df.iterrows(), desc='extracting image'):
            if t2t:
                query_text = row[en_lang_col]
                if t2t_order['model1'] == 'image_text':
                    image_embedding = self.eval_dataset.get_text_embedding(query_text, en_lang_col, use_clip=True)
                else: # multilingual text encoder
                    image_embedding = self.eval_dataset.get_text_embedding(query_text, en_lang_col, use_clip=False)

            else:
                image_url = row[self.eval_dataset.img_col]
                image_embedding = self.eval_dataset.get_image_embedding(image_url)

            all_imgs.append(image_embedding)

        # I and T are same for now
        # I x D
        all_imgs = torch.cat(all_imgs, dim=0).to(self.device)
        all_imgs = torch.nn.functional.normalize(all_imgs, dim=-1, p=2)

        
        if self.args.save_feats:
            save_feat_dir = save_feat_dir or os.path.join(self.args.save_dir,"features")
            os.makedirs(save_feat_dir, exist_ok=True)

            save_p = os.path.join(save_feat_dir, "all_images.pth")
            torch.save(all_imgs.detach().cpu(), save_p)

        t2t_prefix = ""
        if t2t:
            t2t_prefix = f"{t2t_order['model1']}->{t2t_order['model2']}/"
        

        for lang in self.eval_dataset.lang_cols:

            print(f'Evaluating for lang: {lang}')
            all_text = []
            all_text_org = []
            all_text_clip = []
            
            for idx, row in tqdm(self.eval_dataset.img_text_df.iterrows(), desc=f'extracting {lang}'):

                query_text = row[lang]  # Assuming 'caption' is the text query column

                if t2t:
                    if t2t_order['model2'] == 'image_text':
                        query_embedding = self.eval_dataset.get_text_embedding(query_text, lang, use_clip=True).to(self.device)
                    else: # use multilingual text encoder
                        query_embedding = self.eval_dataset.get_text_embedding(query_text, lang, use_clip=False).to(self.device)
                else:
                    query_embedding = self.eval_dataset.get_text_embedding(query_text, lang, use_clip=use_clip).to(self.device)

                # skip model inference for
                # clip and labse baselines

                all_text_org.append(query_embedding.detach().cpu())
                if use_clip or labse_baseline or skip_inference:
                    pass
                else:
                    query_embedding = self.model.inference(query_embedding).detach().cpu()

                all_text.append(query_embedding)



                if self.args.save_feats and lang in ['XTD10_captions_en', 'caption']:
                    # save original clip text features                    
                    clip_text_embedding = self.eval_dataset.get_text_embedding(query_text, lang, use_clip=True)
                    all_text_clip.append(clip_text_embedding.detach().cpu())

            # T x D
            all_text = torch.cat(all_text, dim=0).to(self.device)
            all_text = torch.nn.functional.normalize(all_text, dim=-1, p=2)
                
             
            if self.args.save_feats:
                save_feat_dir_lang = os.path.join(save_feat_dir, lang)
                os.makedirs(save_feat_dir_lang, exist_ok=True)
                
                all_text_org = torch.cat(all_text_org, dim=0).to(self.device)
           
                save_p = os.path.join(save_feat_dir_lang, "all_text_org.pth")
                torch.save(all_text_org.detach().cpu(), save_p)

                save_p = os.path.join(save_feat_dir_lang, "all_text.pth")
                torch.save(all_text.detach().cpu(), save_p)

                if all_text_clip:
                    all_text_clip = torch.cat(all_text_clip, dim=0).to(self.device)
                    save_p = os.path.join(save_feat_dir_lang, "all_text_clip.pth")
                    torch.save(all_text_clip.detach().cpu(), save_p)

            
            N = all_imgs.shape[0]
            D = all_imgs.shape[1]
            

            # old sim
            sim = all_imgs @ all_text.permute(1, 0)
            
            sim = sim.detach().cpu()
            i2t = sim
            t2i = i2t.permute(1,0)

            max_k = max(self.k_values)
            i2t_ranks = torch.argsort(i2t, descending=True, dim=1)[:, :max_k].detach().cpu()
            t2i_ranks = torch.argsort(t2i, descending=True, dim=1)[:, :max_k].detach().cpu()


            if return_top_idx:
                pred_ranks[lang] = {
                    'i2t': i2t_ranks,
                    't2i': t2i_ranks
                }

            i2t_gt_rank = torch.arange(i2t_ranks.shape[0])
            t2i_gt_rank = torch.arange(t2i_ranks.shape[0])

            i2t_is_correct_in_top_k = (i2t_ranks == i2t_gt_rank[:, None])  # Shape: [T, K]
            t2i_is_correct_in_top_k = (t2i_ranks == t2i_gt_rank[:, None])  # Shape: [T, K]

            # Now, compute Recall@K for each K
            i2t_recall = {f'eval/R@{k}/{t2t_prefix}i2t/{lang}': (i2t_is_correct_in_top_k[:, :k].sum(dim=1) > 0).float().mean().item()*100 for k in self.k_values}
            t2i_recall = {f'eval/R@{k}/{t2t_prefix}t2i/{lang}': (t2i_is_correct_in_top_k[:, :k].sum(dim=1) > 0).float().mean().item()*100 for k in self.k_values}

            metrics.update(i2t_recall)
            metrics.update(t2i_recall)
            
            print(i2t_recall)
            #print("Max length 64 count", self.eval_dataset.max_length_64)
            self.eval_dataset.max_length_64 = 0
            print("-"*100)
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

