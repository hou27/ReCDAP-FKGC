import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion import ConditionalUnet1D, DDPMScheduler
from relational_path_gnn import RelationalPathGNN

###################################################
# 1) LSTM_attn : relation_learner (from prior work)
###################################################
class LSTM_attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=1, dropout=0.5):
        super(LSTM_attn, self).__init__()
        self.embed_size = embed_size
        self.n_hidden = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(
            input_size=self.embed_size * 2,
            hidden_size=self.n_hidden,
            num_layers=self.layers,
            bidirectional=True,
            dropout=self.dropout,
        )
        self.out = nn.Linear(self.n_hidden * 2 * self.layers, self.out_size)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, self.n_hidden * 2, self.layers)
        attn_weight = torch.bmm(lstm_output, hidden).squeeze(2).to(lstm_output.device)
        soft_attn_weight = F.softmax(attn_weight, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weight)
        context = context.view(-1, self.n_hidden * 2 * self.layers)
        return context

    def forward(self, inputs):
        size = inputs.shape
        inputs = inputs.contiguous().view(size[0], size[1], -1)
        input = inputs.permute(1, 0, 2)
        hidden_state = torch.zeros(self.layers * 2, size[0], self.n_hidden, device=inputs.device)
        cell_state = torch.zeros(self.layers * 2, size[0], self.n_hidden, device=inputs.device)
        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))  # LSTM
        output = output.permute(1, 0, 2)
        attn_output = self.attention_net(output, final_cell_state)

        outputs = self.out(attn_output)
        return outputs.view(size[0], 1, 1, self.out_size)

###################################################
# 2) ScoreCalculator
###################################################
class ScoreCalculator(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.head_encoder = nn.Linear(4 * emb_dim, emb_dim)
        self.tail_encoder = nn.Linear(4 * emb_dim, emb_dim)

    def forward(self, h, t, r, pos_num, z):
        """
        h, t, r : (B, nq+nn, 1, emb_dim)
        z       : (B, nq+nn, embed_dim)  -> user code에서는 (B, embed_dim)을 unsqueeze
        pos_num : int, query 개수
        return: p_score (B, pos_num), n_score (B, neg_num)
        """
        z_unsq = z.unsqueeze(2)  # (B, nq+nn, 1, embed_dim)
        
        # 머리/꼬리 임베딩을 투영
        h = h + self.head_encoder(z_unsq)
        t = t + self.tail_encoder(z_unsq)
        
        # L2 norm
        score = -torch.norm(h + r - t, p=2, dim=-1)  # (B, nq+nn)
        
        p_score = score[:, :pos_num]
        n_score = score[:, pos_num:]
        return p_score, n_score


###################################################
# 3) Attention Pooler
###################################################
class AttentionPooler(nn.Module):
    def __init__(self, embed_dim=100, num_heads=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1)
        )

        # initialize
        nn.init.xavier_normal_(self.query)

    def forward(self, x):
        """
        x: (B, L, D)
        return: (B, D)
        """
        B, L, D = x.shape
        q = self.query.expand(B, -1, -1)   # (B, 1, D)
        
        attn_output, attn_weights = self.attn(q, x, x)
        out = self.mlp(attn_output)       # (B, 1, D)
        return out.squeeze(1)            # (B, D)


###################################################
# 4) Relation-based Conditional Diffusion with Attention Pooling
###################################################
class ReCDAP(nn.Module):
    def __init__(self, g, dataset, parameter):
        super().__init__()
        
        self.device = parameter['device']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.few = parameter['few']

        self.r_path_gnn = RelationalPathGNN(
            g, 
            dataset['ent2id'], 
            len(dataset['rel2emb']), 
            parameter
        )

        self.relation_learner =  LSTM_attn(embed_size=self.embed_dim, n_hidden=parameter['lstm_hiddendim'], out_size=self.embed_dim,
                                              layers=parameter['lstm_layers'], dropout=self.dropout_p)

        self.score_calculator = ScoreCalculator(self.embed_dim)

        self.attn_pooler = AttentionPooler(embed_dim=self.embed_dim * 2, num_heads=1)
        
        num_diffusion_iters = parameter['num_diffusion_iters']
        self.noise_pred_net = ConditionalUnet1D(input_dim=self.embed_dim * 2, global_cond_dim=self.embed_dim * 1 + self.embed_dim * self.few * 2 * 2 + self.few * 2) # relation + support + pos/neg label
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        
        # MarginRankingLoss
        self.loss_func = nn.MarginRankingLoss(self.margin)

    def split_concat(self, positive, negative):
        """
        positive, negative shape: (B, n, 2, embed_dim)
        -> (B, n+n, 1, embed_dim) for head & tail
        """
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], dim=1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], dim=1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def eval_reset(self):
        self.eval_query = None
        self.eval_target_r = None
        self.eval_rel = None
        self.is_reset = True
    
    def eval_support(self, support, support_negative, query):
        support, support_negative, query = self.r_path_gnn(support), self.r_path_gnn(support_negative), self.r_path_gnn(query)
        B = support.shape[0]
        support_few = support.view(support.shape[0], self.few, 2, self.embed_dim)
        rel = self.relation_learner(support_few)
        support_pos_r = support.view(B, self.few, -1)
        support_neg_r = support_negative.view(B, self.few, -1)
        target_r = torch.cat([support_pos_r, support_neg_r], dim=1)

        return query, target_r, rel

    def eval_forward(self, task):
        support, support_negative, query, negative = task
        negative = self.r_path_gnn(negative)
        if self.is_reset:
            query, target_r, rel = self.eval_support(support, support_negative, query)
            self.eval_query = query
            self.eval_target_r = target_r
            self.eval_rel = rel
            self.is_reset = False
        else:
            query = self.eval_query
            target_r = self.eval_target_r
            rel = self.eval_rel
        B = negative.shape[0]
        num_q = query.shape[1]  # num of query
        num_n = negative.shape[1]  # num of query negative
        # global_cond
        global_cond = torch.cat([rel.view(B, -1), target_r.view(B, 1, -1).squeeze(1), 
                                 torch.ones(B, self.few, device=self.device), 
                                 torch.zeros(B, self.few, device=self.device)], dim=-1)

        # padding
        target_r = F.pad(target_r, (0, 0, 0, 2), mode='constant', value=0.0)

        # Reverse sampling
        with torch.no_grad():
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            
            noise = torch.randn_like(target_r)
            z_denoised = noise
            
            for t in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    sample=z_denoised,
                    timestep=t,
                    global_cond=global_cond
                )
                z_denoised = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=z_denoised
                ).prev_sample

        # (4) Generate global representation pos/neg
        z_without_pad = z_denoised[:, :self.few * 2, :]
        z_pos = z_without_pad[:, :self.few]  # (B, few, embed_dim * 2)
        z_neg = z_without_pad[:, self.few:]  # (B, few, embed_dim * 2)
        z_pos_r = self.attn_pooler(z_pos)  # (B, embed_dim * 2)
        z_neg_r = self.attn_pooler(z_neg)  # (B, embed_dim * 2)
        z = torch.cat([z_pos_r, z_neg_r], dim=-1)  # (B, embed_dim * 2 * 2)
        z_q = z.unsqueeze(1).expand(-1, num_q + num_n, -1)  # (B, num_q + num_n, embed_dim * 2 * 2)

        # (5) 스코어 계산
        #  pos/neg triple (query/negative) -> head/tail concat
        #  => (B, nq+nn, 1, emb_dim)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)
        
        # rel : (B, 1, 1, embed_dim) -> (B, nq+nn, 1, embed_dim)
        rel_q = rel.expand(-1, num_q + num_n, -1, -1)
        
        p_score, n_score = self.score_calculator(que_neg_e1, que_neg_e2, rel_q, num_q, z_q)
        
        return p_score, n_score

    def forward(self, task, iseval=False, istest=False):
        """
        task: (support, support_negative, query, negative)
        각 shape -> self.r_path_gnn -> (B, few, 2, embed_dim) or (B, nq, 2, embed_dim)
        """
        support, support_negative, query, negative = [self.r_path_gnn(t) for t in task]
        B = support.shape[0]
        num_q = query.shape[1]
        num_n = negative.shape[1]

        # relation 임베딩 (support_few)
        support_few = support.view(B, self.few, 2, self.embed_dim)
        rel = self.relation_learner(support_few)  # (B, 1, 1, embed_dim) shape
        
        # support_pos_r, support_neg_r : (B, few, embed_dim * 2)
        support_pos_r = support.view(B, self.few, -1)
        support_neg_r = support_negative.view(B, self.few, -1)

        # pos_r -> (B, few, embed_dim * 2)
        pos_r = support_pos_r
        # neg_r -> (B, few, embed_dim * 2)
        neg_r = support_neg_r
        target_r = torch.cat([pos_r, neg_r], dim=1)  # (B, 2 * few, embed_dim * 2)
        
        # global_cond
        global_cond = torch.cat([rel.view(B, -1), target_r.view(B, 1, -1).squeeze(1), 
                                 torch.ones(B, self.few, device=self.device), 
                                 torch.zeros(B, self.few, device=self.device)], dim=-1)

        # padding
        target_r = F.pad(target_r, (0, 0, 0, 2), mode='constant', value=0.0)

        if istest or iseval:
            pass
        else:            
            # Diffusion

            # Train noise prediction network
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (B,), device=self.device
            ).long()
            noise = torch.randn_like(target_r, device=self.device)
            noisy_z = self.noise_scheduler.add_noise(target_r, noise, timesteps)
            noise_pred = self.noise_pred_net(
                sample=noisy_z,
                timestep=timesteps,
                global_cond=global_cond
            )
            mse_loss = F.mse_loss(noise_pred, noise)

        # Reverse sampling
        with torch.no_grad():
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            
            noise = torch.randn_like(target_r)
            z_denoised = noise
            
            for t in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    sample=z_denoised,
                    timestep=t,
                    global_cond=global_cond
                )
                z_denoised = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=z_denoised
                ).prev_sample

        # (4) Generate global representation pos/neg
        z_without_pad = z_denoised[:, :self.few * 2, :]
        z_pos = z_without_pad[:, :self.few]  # (B, few, embed_dim * 2)
        z_neg = z_without_pad[:, self.few:]  # (B, few, embed_dim * 2)
        z_pos_r = self.attn_pooler(z_pos)  # (B, embed_dim * 2)
        z_neg_r = self.attn_pooler(z_neg)  # (B, embed_dim * 2)
        z = torch.cat([z_pos_r, z_neg_r], dim=-1)  # (B, embed_dim * 2 * 2)
        z_q = z.unsqueeze(1).expand(-1, num_q + num_n, -1)  # (B, num_q + num_n, embed_dim * 2 * 2)

        # (5) 스코어 계산
        #  pos/neg triple (query/negative) -> head/tail concat
        #  => (B, nq+nn, 1, emb_dim)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)
        
        # rel : (B, 1, 1, embed_dim) -> (B, nq+nn, 1, embed_dim)
        rel_q = rel.expand(-1, num_q + num_n, -1, -1)
        
        p_score, n_score = self.score_calculator(que_neg_e1, que_neg_e2, rel_q, num_q, z_q)

        if iseval or istest:
            return p_score, n_score
        else:
            return p_score, n_score, mse_loss