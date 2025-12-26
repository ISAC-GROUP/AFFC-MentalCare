import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Loss Function
class DiffLoss(nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean 

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach() 
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CMD(nn.Module):
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)  
        return scms 

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, original, reconstructed, d):
        batch_size = original.size(0)
        mse = torch.mean((original - reconstructed) ** 2, dim=1)  
        loss = mse
        return torch.mean(loss)  

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)

        # L2 normalize
        features = F.normalize(features, dim=1)

        # similarity matrix: [B, B]
        similarity_matrix = torch.matmul(features, features.t()) / self.temperature

        # mask for same subject
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(device)
        mask = mask - torch.eye(batch_size, device=device)  

        # --------- log-sum-exp trick ---------
        sim_max, _ = similarity_matrix.max(dim=1, keepdim=True)  # [B,1]
        exp_sim = torch.exp(similarity_matrix - sim_max) * (1 - torch.eye(batch_size, device=device))
        log_prob = similarity_matrix - sim_max - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)

        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)  
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum

        loss = -mean_log_prob_pos.mean()
        return loss
        
class PPG_IR_IRED_Encoder(nn.Module):
    def __init__(self, batch_size=128, seq_len=400, n_channels=1):
        super(PPG_IR_IRED_Encoder, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_channels = n_channels

        # Layer 1 - Conv2D + BatchNorm
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 25),   
                padding=(0, 12),      
                bias=False),
            nn.BatchNorm2d(F1)
        )

        # Layer 2 - DepthwiseConv2D + AvgPool2D
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=F1,
                      out_channels=F1,
                      kernel_size=(1, 7),
                      groups=F1,         
                      bias=False),
            nn.ELU(),
            nn.Conv2d(in_channels=F1,
                      out_channels=F1 * D,
                      kernel_size=(1, 1), 
                      bias=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),  
            nn.Dropout(p=0.5)
        )

        # Layer 3 - SeparableConv2D + AvgPool2D
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 15),  
                      padding=(0, 7),
                      groups=F1 * D,
                      bias=False),
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 1),  
                      bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)), 
            nn.Dropout(p=0.5)
        )

    def get_feature(self, x):
        x = self.block1(x)  # [bs, 1, 1, 400] -> [bs, 8, 1, 400]
        x = self.block2(x)  # -> [bs, 16, 1, 200]
        x = self.block3(x)  # -> [bs, 16, 1, 100]
        return x

    def forward(self, x):
        x = x.reshape(-1, 1, self.n_channels, self.seq_len)
        x = self.get_feature(x)
        x = x.view(x.size(0), -1)
        return x


class SKT_Encoder(nn.Module):
    def __init__(self, batch_size=128, seq_len=100, n_channels=1):
        super(SKT_Encoder, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_channels = n_channels

        # Layer 1 - Conv2D + BatchNorm
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,           
                out_channels=F1,
                kernel_size=(1, 25),     
                padding=(0, 12),         
                bias=False),
            nn.BatchNorm2d(F1)
        )

        # Layer 2 - DepthwiseConv2D + AvgPool2D
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=F1,
                      out_channels=F1,
                      kernel_size=(1, 5),  
                      groups=F1,          
                      bias=False),
            nn.ELU(),
            nn.Conv2d(in_channels=F1,
                      out_channels=F1 * D,
                      kernel_size=(1, 1),  
                      bias=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)), 
            nn.Dropout(p=0.5)
        )

        # Layer 3 - SeparableConv2D + AvgPool2D
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 15),  
                      padding=(0, 7),
                      groups=F1 * D,
                      bias=False),
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 1),  
                      bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )

    def get_feature(self, x):
        x = self.block1(x)  # [bs, 1, 1, 100] -> [bs, 8, 1, 100]
        x = self.block2(x)  # -> [bs, 16, 1, 50]
        x = self.block3(x)  # -> [bs, 16, 1, 25]
        return x

    def forward(self, x):
        x = x.reshape(-1, 1, self.n_channels, self.seq_len)
        x = self.get_feature(x)
        x = x.view(x.size(0), -1)
        return x

class GSR_Encoder(nn.Module):
    def __init__(self, batch_size=512, seq_len=200, n_channels=1):
        super(GSR_Encoder, self).__init__()
        F1, D = 8, 2
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n_channels = n_channels

        # Layer 1 - Conv2D + BatchNorm
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, 25),    
                padding=(0, 12),        
                bias=False),
            nn.BatchNorm2d(F1)
        )

        # Layer 2 - DepthwiseConv2D + AvgPool2D
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=F1,
                      out_channels=F1,
                      kernel_size=(1, 5), 
                      groups=F1,            
                      bias=False),
            nn.ELU(),
            nn.Conv2d(in_channels=F1,
                      out_channels=F1 * D,
                      kernel_size=(1, 1),   
                      bias=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),  
            nn.Dropout(p=0.5)
        )

        # Layer 3 - SeparableConv2D + AvgPool2D
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 15), 
                      padding=(0, 7),
                      groups=F1 * D,      
                      bias=False),
            nn.Conv2d(in_channels=F1 * D,
                      out_channels=F1 * D,
                      kernel_size=(1, 1),  
                      bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 2)),  
            nn.Dropout(p=0.5)
        )

    def get_feature(self, x):
        x = self.block1(x)  # [bs, 1, 1, 200] -> [bs, 8, 1, 200]
        x = self.block2(x)  # -> [bs, 16, 1, 100]
        x = self.block3(x)  # -> [bs, 16, 1, 50]
        return x

    def forward(self, x):
        x = x.reshape(-1, 1, self.n_channels, self.seq_len)
        x = self.get_feature(x)
        x = x.view(x.size(0), -1)
        return x

class CMAI(nn.Module):
    def __init__(self, hidden_size, num_modalities=5):
        super(CMAI, self).__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities  

        # ----- Phase 1: Cross-modal interaction -----
        self.Wk_so = nn.Linear(num_modalities * hidden_size, num_modalities * hidden_size)
        self.Wv_so = nn.Linear(num_modalities * hidden_size, num_modalities * hidden_size)
        self.norm_so = nn.LayerNorm(num_modalities * hidden_size)

        # Q_i : [B, H] -> [B, H]
        self.Wq_ta = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_modalities)])
        self.norm_ta = nn.LayerNorm(hidden_size)

        # [B, H]
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # ----- Phase 2: Adaptive attention -----
        self.q = nn.Parameter(torch.randn(hidden_size, 1))  # q:[H,1]
        self.attn_W = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_modalities)])
        self.attn_b = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, 1)) for _ in range(num_modalities)])

    def forward(self, x):
        M, B, H = x.shape  

        # ------ Phase 1: Cross-modal interaction ------
        F_so = x.transpose(0, 1)              # [B, M, H]
        F_flat = F_so.reshape(B, M * H)       # [B, M*H]

        K_so = self.Wk_so(self.norm_so(F_flat)).view(B, M, H)  # [B, M, H]
        V_so = self.Wv_so(self.norm_so(F_flat)).view(B, M, H)  # [B, M, H]

        outputs = []
        for i in range(M):
            F_ta = x[i]                             # [B, H]
            Q_ta = self.Wq_ta[i](self.norm_ta(F_ta))  # [B, H]

            # [B, 1, H] @ [B, H, M] -> [B, 1, M]
            attn_w = torch.bmm(Q_ta.unsqueeze(1), K_so.transpose(1, 2))
            attn_w = torch.softmax(attn_w, dim=-1)      # [B, 1, M]

            # [B, 1, M] @ [B, M, H] -> [B, 1, H] -> [B, H]
            attn = torch.bmm(attn_w, V_so).squeeze(1)   # [B, H]

            Y_ta = self.norm_ta(F_ta) + attn            # [B, H]
            Y_ta = self.ffn(self.norm_ta(Y_ta)) + Y_ta  # [B, H]
            outputs.append(Y_ta)

        # ------ Phase 2: Adaptive attention ------
        scores = []
        for i in range(M):
            Y = outputs[i]                                   # [B, H]
            s = torch.tanh(self.attn_W[i](Y) + self.attn_b[i].T)  # [B, H]
            mu = (s @ self.q).squeeze(-1)                    # [B] q^T * tanh(...)
            scores.append(mu)

        weights = torch.softmax(torch.stack(scores, dim=1), dim=1)  # [B, M]
        Y_stack = torch.stack(outputs, dim=1)                       # [B, M, H]
        fused = torch.sum(weights.unsqueeze(-1) * Y_stack, dim=1)   # [B, H]
        return fused




class DomainMapper(nn.Module):
    def __init__(self, in_dim=640, hidden_dim=256, num_train_sources=32):
        super(DomainMapper, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_train_sources)  
        )

    def forward(self, x, subject_labels):
        unique_ids = torch.unique(subject_labels)
        subj_feats = []
        for sid in unique_ids:
            mask = (subject_labels == sid)
            subj_feat = x[mask].mean(dim=0)  # [in_dim]
            subj_feats.append(subj_feat)
        subj_feats = torch.stack(subj_feats, dim=0)  # [num_subjects_in_batch, in_dim]

        logits = self.mlp(subj_feats)  # [num_subjects_in_batch, num_sources]
        probs = F.softmax(logits, dim=-1)  
        return probs, unique_ids

#  Main Class
class CDPT(nn.Module):
    def __init__(self, config, test_mode):
        super(CDPT, self).__init__()

        self.config = config
        self.tau = self.config.tau  # balance the weight of general and privacy classification
        self.topM = self.config.topM
        self.subject_num = subject_num = config.subject_num
        self.hidden_size = hidden_size = config.hidden_size
        self.output_size = output_size = config.num_classes
        self.PPGFeatureExtractor = PPG_IR_IRED_Encoder(batch_size=self.config.batch_size, seq_len=400, n_channels=1)
        self.IRFeatureExtractor = PPG_IR_IRED_Encoder(batch_size=self.config.batch_size, seq_len=400, n_channels=1)
        self.IRedFeatureExtractor = PPG_IR_IRED_Encoder(batch_size=self.config.batch_size, seq_len=400, n_channels=1)
        self.GSRFeatureExtractor = GSR_Encoder(batch_size=self.config.batch_size, seq_len=200, n_channels=1)
        self.SKTFeatureExtractor = SKT_Encoder(batch_size=self.config.batch_size, seq_len=100, n_channels=1)
        self.AdaptiveWeightCalculator = DomainMapper(in_dim=config.hidden_size, hidden_dim=self.config.hidden_size, num_train_sources=self.subject_num)
        self.subject_contrastive_loss = ContrastiveLoss(temperature=0.1)
        self.dropout_rate = config.dropout
        self.activation = self.config.activation
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.ELU = nn.ELU()
        self.test_mode = test_mode

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        self.project_ppg = nn.Sequential()
        self.project_ppg.add_module('project_ppg', nn.Linear(in_features=1568, out_features=config.hidden_size))
        self.project_ppg.add_module('project_ppg_activation', self.activation()) # nn.ReLU()
        self.project_ppg.add_module('project_ppg_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_ir = nn.Sequential()
        self.project_ir.add_module('project_ir', nn.Linear(in_features=1568, out_features=config.hidden_size))
        self.project_ir.add_module('project_ir_activation', self.activation()) # nn.ReLU()
        self.project_ir.add_module('project_ir_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_ired = nn.Sequential()
        self.project_ired.add_module('project_ired', nn.Linear(in_features=1568, out_features=config.hidden_size))
        self.project_ired.add_module('project_ired_activation', self.activation()) # nn.ReLU()
        self.project_ired.add_module('project_ired_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_gsr = nn.Sequential()
        self.project_gsr.add_module('project_gsr', nn.Linear(in_features=784, out_features=config.hidden_size))
        self.project_gsr.add_module('project_gsr_activation', self.activation()) # nn.ReLU()
        self.project_gsr.add_module('project_gsr_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_skt = nn.Sequential()
        self.project_skt.add_module('project_skt', nn.Linear(in_features=384, out_features=config.hidden_size))
        self.project_skt.add_module('project_skt_activation', self.activation()) # nn.ReLU()
        self.project_skt.add_module('project_skt_layer_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # subject private encoders
        ##########################################
        
        self.subject_private = []
        for i in range(subject_num):
                self.subject_private.append(nn.Sequential(
                    nn.Linear(in_features=hidden_size, out_features=hidden_size),
                    self.ELU,
                    nn.Dropout(p=self.dropout_rate),
                    nn.Linear(in_features=hidden_size, out_features=hidden_size)
                ).to(self.config.device))
        

        ##########################################
        # subject shared encoder
        ##########################################

        self.subject_shared = nn.Sequential()
        self.subject_shared.add_module('shared_1', nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.subject_shared.add_module('shared_activation_1', self.ELU)
        self.subject_shared.add_module('Dropout', nn.Dropout(p=self.dropout_rate))
        self.subject_shared.add_module('shared_2', nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.subject_shared.add_module('shared_batch_norm_2', nn.BatchNorm1d(hidden_size))

        ##########################################
        # subject reconstruct
        ##########################################

        self.recon_subject = nn.Sequential()
        self.recon_subject.add_module('recon', nn.Linear(in_features=hidden_size, out_features=hidden_size))

        self.cross_modal_attn = CMAI(hidden_size=self.hidden_size, num_modalities=5)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, dim_feedforward=hidden_size*2,nhead=2,dropout=self.dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.utt_feature_norm = nn.BatchNorm1d(self.hidden_size)

        ##########################################
        # heterogenity subject feature positive transfer
        # positive transfer adapter
        # multiple classifier 
        ##########################################

        self.multiple_heterogeneous_classifiers = []
        for i in range(subject_num):
                self.multiple_heterogeneous_classifiers.append(nn.Sequential(
                    nn.Linear(in_features=hidden_size, out_features=hidden_size),
                    self.ELU,
                    nn.Dropout(p=self.dropout_rate),
                    nn.Linear(in_features=hidden_size, out_features=self.output_size)
                ).to(self.config.device))

        self.general_classifiers = nn.Sequential()
        self.general_classifiers.add_module('general_linear_1', nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.general_classifiers.add_module('general_activation', nn.ELU())
        self.general_classifiers.add_module('general_dropout', nn.Dropout(p=self.dropout_rate))
        self.general_classifiers.add_module('general_linear_2', nn.Linear(in_features=hidden_size, out_features=self.output_size))

    def get_topM_values_and_indices(self, M):
        """
        return:
            topM_values:  [num_unique_subjects, M]
            topM_indices: [num_unique_subjects, M]
            unique_ids:   [num_unique_subjects]
        """
        topM_values, topM_indices = torch.topk(self.adaptive_weight, k=M, dim=1)

        return topM_values, topM_indices

    def extract_features(self, extractor, x):
        features = extractor(x) 
        return features


    def alignment(self, ppg, ir, ired, gsr, skt, subject_labels, test_mode):
        batch_size = self.config.batch_size
        self.ppg = ppg
        self.ir = ir
        self.ired = ired
        self.gsr = gsr
        self.skt = skt

        # extract features from physiological modality
        self.utterance_ppg = self.extract_features(self.PPGFeatureExtractor, self.ppg)
        self.utterance_ir = self.extract_features(self.IRFeatureExtractor, self.ir)
        self.utterance_ired = self.extract_features(self.IRedFeatureExtractor, self.ired)
        self.utterance_gsr = self.extract_features(self.GSRFeatureExtractor, self.gsr)
        self.utterance_skt = self.extract_features(self.SKTFeatureExtractor, self.skt)

        self.utterance_ppg = self.project_ppg(self.utterance_ppg)
        self.utterance_ir = self.project_ir(self.utterance_ir)
        self.utterance_ired = self.project_ired(self.utterance_ired)
        self.utterance_gsr = self.project_gsr(self.utterance_gsr)
        self.utterance_skt = self.project_skt(self.utterance_skt)

        # # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utterance_ppg, self.utterance_ir, self.utterance_ired, self.utterance_gsr, self.utterance_skt), dim=0)
        h = self.cross_modal_attn(h)  # [5, BS, H]
        self.h = self.utt_feature_norm(h)

        if test_mode:
            self.utt_shared_subject = self.subject_shared(self.h)
            self.utt_private_subject = []

            self.adaptive_weight, self.unique_ids = self.AdaptiveWeightCalculator(self.h, subject_labels)
            self.topM_values, self.topM_indices = self.get_topM_values_and_indices(self.topM)  # [unique_id_num=1, topM]
            # print('self.topM_values, self.topM_indices:', self.topM_values, self.topM_indices)

            self.cla_pub_out = self.general_classifiers(self.utt_shared_subject)
            self.cla_pri_out = []
            topM_output = []
            for k, i in enumerate(self.topM_indices[0]):
                encoder = self.subject_private[i]
                heter_feature = encoder(self.h)  # [BS, 640]
                classifier = self.multiple_heterogeneous_classifiers[i]
                het_out = classifier(heter_feature)  # [BS, num_classes]
                weight = self.topM_values[0, k]  # scalar
                weighted_het_out = het_out * weight  # [BS, num_classes]
                topM_output.append(weighted_het_out)

            topM_output = torch.stack(topM_output, dim=0)  # [M, BS, num_classes]
            topM_weighted_output = topM_output.sum(dim=0)  # [BS, num_classes]

            self.cla_pri_out = topM_weighted_output
            balance_output = self.tau * self.cla_pub_out + (1 - self.tau) * self.cla_pri_out
            return balance_output
        else:
            self.utt_shared_subject = self.subject_shared(self.h)
            self.utt_private_subject = []
            for i, sid in enumerate(subject_labels):
                encoder = self.subject_private[int(sid.item())]  
                current_h = self.h[i].reshape(1,-1)
                current_pri_out = encoder(current_h)
                self.utt_private_subject.append(current_pri_out)
            self.utt_private_subject = torch.cat(self.utt_private_subject, dim=0)
            self.utt_subject = self.utt_private_subject + self.utt_shared_subject
            self.utt_subject_recon = self.recon_subject(self.utt_subject)

            self.adaptive_weight, self.unique_ids = self.AdaptiveWeightCalculator(self.h, subject_labels) #[bs, source_sub_num] -> [unique_id_num, source_sub_num]
            self.topM_values, self.topM_indices = self.get_topM_values_and_indices(self.topM) # [unique_id_num, topM]
            self.cla_pub_out = self.general_classifiers(self.utt_shared_subject)
            self.cla_pri_out = []
            for i, sid in enumerate(subject_labels): # i is subject_id
                idx = torch.where(self.unique_ids == sid)[0]
                topM_ids = self.topM_indices[idx][0]
                topM_values = self.topM_values[idx]
                topM_output = []
                for j in topM_ids:
                    heter_classifer = self.multiple_heterogeneous_classifiers[j]
                    topM_output.append(heter_classifer(self.utt_subject[i]))
                topM_output = torch.stack(topM_output)
                weights = topM_values.view(-1, 1) 
                weighted_output = topM_output * weights
                weighted_output = weighted_output.sum(dim=0)
                self.cla_pri_out.append(weighted_output)
            self.cla_pri_out = torch.stack(self.cla_pri_out)
            balance_output = self.tau * self.cla_pub_out + (1-self.tau) * self.cla_pri_out
            return balance_output


    def reconstruct(self,):
        self.utt_ppg = (self.utt_private_ppg + self.utt_shared_ppg)
        self.utt_ir = (self.utt_private_ir + self.utt_shared_ir)
        self.utt_ired = (self.utt_private_ired + self.utt_shared_ired)
        self.utt_gsr = (self.utt_private_gsr + self.utt_shared_gsr)
        self.utt_skt = (self.utt_private_skt + self.utt_shared_skt)
        self.utt_ppg_recon = self.recon_ppg(self.utt_ppg)
        self.utt_ir_recon = self.recon_ir(self.utt_ir)
        self.utt_ired_recon = self.recon_ired(self.utt_ired)
        self.utt_gsr_recon = self.recon_gsr(self.utt_gsr)
        self.utt_skt_recon = self.recon_skt(self.utt_skt)

    def compute_subject_losses(self, utt_shared_subject, utt_private_subject, subject_labels):
        d = self.utt_subject.size(1)
        diff_loss = DiffLoss()(utt_shared_subject, utt_private_subject)
        recon_loss = ReconstructionLoss()(self.h, self.utt_subject_recon, d)
        contrastive_loss = self.subject_contrastive_loss(utt_private_subject, subject_labels)
        return diff_loss, recon_loss, contrastive_loss

    def compute_adaptive_weight_loss(self,):
        criterion = nn.NLLLoss()
        adaptive_loss = criterion(torch.log(self.adaptive_weight), self.unique_ids)
        return adaptive_loss

    def forward(self, ppg, ir, ired, gsr, skt, subject_labels, alpha): # alpha is not applicable
        output = self.alignment(ppg, ir, ired, gsr, skt, subject_labels, self.test_mode)
        if self.test_mode == False:
            diff_s_loss, recon_s_loss, contrastive_s_loss = self.compute_subject_losses(
                self.utt_shared_subject, self.utt_private_subject, subject_labels
            )

            adaptive_loss = self.compute_adaptive_weight_loss()

            in_loss_total = (
                self.config.delta * diff_s_loss,
                self.config.epsilon * recon_s_loss,
                self.config.adaptive * adaptive_loss,
                self.config.zeta * contrastive_s_loss
            )
            return in_loss_total, output
        else:
            return output




