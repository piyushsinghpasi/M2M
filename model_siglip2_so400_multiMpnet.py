import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModel
from losses import CUSTOM_LOSS
'''
MAJOR CHANGES:
Added new loss like Labse-En and Labse-Non-EN KL
TO CLIP-En and Labse-En KL
'''

CLIP_MODEL_NAME = "google/siglip2-so400m-patch14-384"
#LABSE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
LABSE_MODEL_NAME = "sentence-transformers/LaBSE"

"""# **Model**"""

class labse_clip(torch.nn.Module, CUSTOM_LOSS):
    def __init__(self, hdim, args):
        super().__init__()
        self.labse = SentenceTransformer(LABSE_MODEL_NAME)
        self.clip_model = AutoModel.from_pretrained(CLIP_MODEL_NAME, trust_remote_code=True)
        self.clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME, trust_remote_code=True)

        self.freeze_encoders()
        self.args = args

        # learnt parameters
        self.mlps = torch.nn.ModuleList([
            torch.nn.Linear(hdim, hdim)
            for _ in range(self.args.num_layers)
        ])

    def freeze_encoders(self):
        for param in self.labse.parameters():
            param.requires_grad = False
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, inputs):

        org_eng_labse_emb = inputs['eng_labse_emb']
        new_eng_labse_emb = org_eng_labse_emb

        # normalize clip text emb
        eng_clip_emb = inputs['eng_clip_emb']
        eng_clip_emb = torch.nn.functional.normalize(eng_clip_emb, dim=-1, p=2)

        new_eng_labse_emb = self.inference(new_eng_labse_emb)
        
        if isinstance(inputs['non_eng_labse_emb'], list):
            new_non_eng_labse_emb = None
        else:
            new_non_eng_labse_emb = self.inference(inputs['non_eng_labse_emb'])
        
        return {
            'non_eng_labse_emb': inputs['non_eng_labse_emb'],
            'new_non_eng_labse_emb': new_non_eng_labse_emb,
            'new_eng_labse_emb': new_eng_labse_emb,
            'org_eng_labse_emb': org_eng_labse_emb,
            'eng_clip_emb': eng_clip_emb,
        }

    
    def inference(self, labse_emb):

        for mlp in self.mlps:
            if self.args.emb_method == "skip_conn":
                labse_emb = labse_emb + mlp(labse_emb)
            else:
                labse_emb = mlp(labse_emb)

        labse_emb = torch.nn.functional.normalize(labse_emb, dim=-1, p=2)

        return labse_emb

    '''
    def mse_loss(self, outputs):
        return torch.nn.functional.mse_loss(outputs['new_eng_labse_emb'], outputs['eng_clip_emb'])
    
    def l1_loss(self, outputs):
        return torch.nn.functional.l1_loss(outputs['new_eng_labse_emb'], outputs['eng_clip_emb'])


    def kl_loss(self, outputs, use_mse=False, use_l1=False):
        # diff batch size is allowed

        # B x D
        new_eng_labse_emb = outputs['new_eng_labse_emb']
        org_eng_labse_emb = outputs['org_eng_labse_emb']

        # B' x D
        non_eng_labse_emb = outputs['non_eng_labse_emb']

        if len(new_eng_labse_emb.shape) == 3 and new_eng_labse_emb.shape[1] == 1:
            new_eng_labse_emb = new_eng_labse_emb.squeeze(1)

        if len(org_eng_labse_emb.shape) == 3 and org_eng_labse_emb.shape[1] == 1:
            org_eng_labse_emb = org_eng_labse_emb.squeeze(1)

        if len(non_eng_labse_emb.shape) == 3 and non_eng_labse_emb.shape[1] == 1:
            non_eng_labse_emb = non_eng_labse_emb.squeeze(1)

        # B x B'
        # predicted distribution
        eps = 1e-8
        pred_sim_score = new_eng_labse_emb @ non_eng_labse_emb.permute(1, 0)
        # pred_sim_score = pred_sim_score.exp().softmax(-1)

        # B x B'
        # true distribution
        gt_sim_score = org_eng_labse_emb @ non_eng_labse_emb.permute(1, 0)
        # gt_sim_score = gt_sim_score.exp().softmax(-1)


        # batchmean reduction in pytorch implementation
        # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        if use_mse:
            kl_loss = torch.nn.functional.mse_loss(pred_sim_score, gt_sim_score)
        elif use_l1:
            kl_loss = torch.nn.functional.l1_loss(pred_sim_score, gt_sim_score)
        else:
            # WARNING: DO NOT USE with COSINE similarity
            kl_loss = (gt_sim_score * (gt_sim_score.log() - pred_sim_score.log())).sum(-1).mean()

        return kl_loss


    def kl_clip_loss(self, outputs, use_mse=False, use_l1=False):
        # diff batch size is allowed

        # B x D
        new_eng_labse_emb = outputs['new_eng_labse_emb']
        clip_eng_emb = outputs['eng_clip_emb']


        if len(new_eng_labse_emb.shape) == 3 and new_eng_labse_emb.shape[1] == 1:
            new_eng_labse_emb = new_eng_labse_emb.squeeze(1)

        if len(clip_eng_emb.shape) == 3 and clip_eng_emb.shape[1] == 1:
            clip_eng_emb = clip_eng_emb.squeeze(1)


        # B x B'
        # predicted distribution
        eps = 1e-8
        pred_sim_score = new_eng_labse_emb @ new_eng_labse_emb.permute(1, 0)
        pred_sim_score = torch.triu(pred_sim_score, diagonal=1)
        # pred_sim_score = pred_sim_score.exp().softmax(-1)

        # B x B'
        # true distribution
        gt_sim_score = clip_eng_emb @ clip_eng_emb.permute(1, 0)
        gt_sim_score = torch.triu(gt_sim_score, diagonal=1)

        
        B = gt_sim_score.size(0)
        num_samples = (B*B - B)/2.0

        # batchmean reduction in pytorch implementation
        # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        if use_mse:
            kl_loss = torch.nn.functional.mse_loss(pred_sim_score, gt_sim_score, reduction='sum')
        elif use_l1:
            kl_loss = torch.nn.functional.l1_loss(pred_sim_score, gt_sim_score, reduction='sum')

        kl_loss = kl_loss / num_samples
        
        return kl_loss


    def new_kl_loss(self, outputs, use_mse=False, use_l1=False):
        # diff batch size is allowed

        # B x D
        new_eng_labse_emb = outputs['new_eng_labse_emb']
        org_eng_labse_emb = outputs['org_eng_labse_emb']

        # B' x D
        non_eng_labse_emb = outputs['non_eng_labse_emb']
        new_non_eng_labse_emb = outputs['new_non_eng_labse_emb']

        if len(new_eng_labse_emb.shape) == 3 and new_eng_labse_emb.shape[1] == 1:
            new_eng_labse_emb = new_eng_labse_emb.squeeze(1)

        if len(org_eng_labse_emb.shape) == 3 and org_eng_labse_emb.shape[1] == 1:
            org_eng_labse_emb = org_eng_labse_emb.squeeze(1)
    
        if len(non_eng_labse_emb.shape) == 3 and non_eng_labse_emb.shape[1] == 1:
            non_eng_labse_emb = non_eng_labse_emb.squeeze(1)
        
        if len(new_non_eng_labse_emb.shape) == 3 and new_non_eng_labse_emb.shape[1] == 1:
            new_non_eng_labse_emb = new_non_eng_labse_emb.squeeze(1)

        # B x B'
        # predicted distribution
        eps = 1e-8
        pred_sim_score = new_eng_labse_emb @ new_non_eng_labse_emb.permute(1, 0)

        # B x B'
        # true distribution
        gt_sim_score = org_eng_labse_emb @ non_eng_labse_emb.permute(1, 0)


        # batchmean reduction in pytorch implementation
        # https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        if use_mse:
            kl_loss = torch.nn.functional.mse_loss(pred_sim_score, gt_sim_score)
        elif use_l1:
            kl_loss = torch.nn.functional.l1_loss(pred_sim_score, gt_sim_score)
        else:
            # WARNING: DO NOT USE with COSINE similarity
            kl_loss = (gt_sim_score * (gt_sim_score.log() - pred_sim_score.log())).sum(-1).mean()

        return kl_loss


    def eng_cosine_loss(self, outputs):
        eng_clip = outputs['eng_clip_emb']
        new_eng_labse_emb = outputs['new_eng_labse_emb']


        if len(new_eng_labse_emb.shape) == 3 and new_eng_labse_emb.shape[1] == 1:
            new_eng_labse_emb = new_eng_labse_emb.squeeze(1)


        if len(eng_clip.shape) == 3 and eng_clip.shape[1] == 1:
            eng_clip = eng_clip.squeeze(1)


        sim = (eng_clip * new_eng_labse_emb).sum(-1)
        loss = (1-sim).mean()

        return loss


    '''
