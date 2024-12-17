import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import pickle
import random


class CrossModalAttention(nn.Module):
    def _init_(self, rec_dim, text_dim):
        super()._init_()
        self.temperature = nn.Parameter(torch.ones(1))
        self.rec_proj = nn.Linear(rec_dim, 256)
        self.text_proj = nn.Linear(text_dim, 256)
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )

    def forward(self, rec_emb, text_emb):
        rec_feat = self.rec_proj(rec_emb)
        text_feat = self.text_proj(text_emb)
        
        attn = torch.matmul(rec_feat, text_feat.transpose(-2, -1)) / self.temperature
        attn = F.softmax(attn, dim=-1)
        
        rec_attended = torch.matmul(attn, text_feat)
        fused = self.fusion(torch.cat([rec_feat, rec_attended], dim=-1))
        return fused


class PointWiseFeedForward(torch.nn.Module):
    def _init_(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self)._init_()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRec_CTRL(torch.nn.Module):
    def _init_(self, user_num, item_num, args):
        super(SASRec_CTRL, self)._init_()
        self.kwargs = {'user_num': user_num, 'item_num': item_num, 'args': args}
        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.description = args.use_description

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.sbert = SentenceTransformer('nq-distilbert-base-v1')
        self.cross_modal = CrossModalAttention(args.hidden_units, 768)
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.final_layer = torch.nn.Linear(args.hidden_units, args.hidden_units)
        
        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        
        self.args = args
        self.bce_criterion = torch.nn.BCEWithLogitsLoss()

        with open(f'./data/Movies_and_TV_meta.json.gz', 'rb') as ft:
            self.text_name_dict = pickle.load(ft)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(
                args.hidden_units,
                args.num_heads,
                args.dropout_rate
            )
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def find_item_text(self, item, title_flag=True, description_flag=True):
        t = 'title'
        d = 'description'
        t_ = 'No Title'
        d_ = 'No Description'
        if title_flag and description_flag:
            return [f'"Title:{self.text_name_dict[t].get(i,t_)}, Description:{self.text_name_dict[d].get(i,d_)}"' for i in item]
        elif title_flag and not description_flag:
            return [f'"Title:{self.text_name_dict[t].get(i,t_)}"' for i in item]
        elif not title_flag and description_flag:
            return [f'"Description:{self.text_name_dict[d].get(i,d_)}"' for i in item]

    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs = self.item_emb.embedding_dim * 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs, mode='default', pretrain=True, opt=None):
        if pretrain:
            log_feats = self.log2feats(log_seqs)
            total_loss = 0
            iterss = 0
            log_feats = log_feats.reshape(-1, log_feats.shape[2])
            log_feats = log_feats[log_seqs.reshape(log_seqs.size)>0]
            text_list = []
            
            for l in log_seqs:
                ll = l[l>0]
                for i in range(len(ll)):
                    to_text = ll[:i+1]
                    text = "This is a user, who has recently watched " + '|'.join(self.find_item_text(to_text, description_flag=False))
                    text += '. This is a movie, title is ' + ','.join(self.find_item_text(to_text, description_flag=self.description))
                    text_list.append(text)

            token = self.sbert.tokenize(text_list)
            text_embedding = self.sbert({
                'input_ids': token['input_ids'].to(log_feats.device),
                'attention_mask': token['attention_mask'].to(log_feats.device)
            })['sentence_embedding']
            ''''
            # Original latent space matching implementation
            text_embedding = self.projection(text_embedding)
            log_feats = self.projection2(log_feats)
            
            start_idx = 0
            end_idx = 32
            loss = 0
            
            while start_idx <len(text_embedding):
                cal = 0
                log = log_feats[start_idx:end_idx]
                text_ = text_embedding[start_idx:end_idx]
                start_idx+=32
                end_idx +=32
                iterss +=1
                
                log_fine1 = self.finegrain1_1(log)
                log_fine2 = self.finegrain1_2(log)
                log_fine3 = self.finegrain1_3(log)
                
                text_fine1 = self.finegrain2_1(text_)
                text_fine2 = self.finegrain2_2(text_)
                text_fine3 = self.finegrain2_3(text_)
                
                sim_mat1 = torch.matmul(log_fine1, text_fine1.T).unsqueeze(0)
                sim_mat2 = torch.matmul(log_fine1, text_fine2.T).unsqueeze(0)
                sim_mat3 = torch.matmul(log_fine1, text_fine3.T).unsqueeze(0)
                
                results1 = torch.cat([sim_mat1,sim_mat2,sim_mat3],dim=0).max(axis=0)[0]
            '''
            start_idx = 0
            end_idx = 32
            loss = 0

            while start_idx < len(text_embedding):
                text_batch = text_embedding[start_idx:end_idx]
                log_batch = log_feats[start_idx:end_idx]
                start_idx += 32
                end_idx += 32
                iterss += 1

                fused_features = self.cross_modal(log_batch, text_batch)
                similarity = torch.matmul(fused_features, text_batch.transpose(-2, -1))
                similarity = similarity / self.temperature

                pos_labels = torch.ones(similarity.diag().shape, device=log_feats.device)
                neg_labels = torch.zeros(similarity[~torch.eye(len(text_batch),dtype=bool)].shape, device=log_feats.device)

                cal = self.bce_criterion(similarity.diag(), pos_labels)
                cal += self.bce_criterion(similarity[~torch.eye(len(similarity),dtype=bool)], neg_labels)

                loss += cal

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            return total_loss

        else:
            log_feats = self.log2feats(log_seqs)
            log_feats = self.final_layer(log_feats)

            if mode == 'log_only':
                log_feats = log_feats[:, -1, :]
                return log_feats

            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            neg_logits = (log_feats * neg_embs).sum(dim=-1)

            if self.args.pretrain_stage:
                return (
                    log_feats.reshape(-1, log_feats.shape[2]),
                    pos_embs.reshape(-1, log_feats.shape[2]),
                    neg_embs.reshape(-1, log_feats.shape[2])
                )
            else:
                return pos_logits, neg_logits

    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs)
        log_feats = self.final_layer(log_feats)
        
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        
        return logits