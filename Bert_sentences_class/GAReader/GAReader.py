# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
from transformers import XLNetModel

from Bert_sentences_class.Models.UnOrderedLSTM import LSTM
from Bert_sentences_class.Models.Linear import Linear
from Bert_sentences_class.Models.MLPAttention import MLPAttention
from Bert_sentences_class.Models.BertEmbedding import BertEmb,Qa_attention_Emb,OO_interaction


# def BiMatching(article,question): # ariticle,question [batch * hidden]


class GAReader(nn.Module):


    # def __init__(self, embedding_dim, output_dim, hidden_size, rnn_num_layers, ga_layers, bidirectional, dropout, word_emb):
    def __init__(self, embedding_dim, output_dim, hidden_size, rnn_num_layers, ga_layers, bidirectional, dropout, bert_config):
        super(GAReader, self).__init__()
        self.word_embedding = BertEmb(bert_config)

        # self.BiMatching = Qa_attention_Emb(embedding_dim)
        self.oo_attention = OO_interaction(embedding_dim,output_dim)
        self.final_liear = nn.Linear(embedding_dim * 3, 1)
        self.softmax = nn.Softmax(dim = 1)
        # self.batch_norm = nn.BatchNorm1d() 
        encoder_layer = TransformerEncoderLayer(768,nhead=8,dropout=0.1)
        encoder_norm = nn.LayerNorm(768)
        self.encoder = TransformerEncoder(
            encoder_layer,
            4,
            encoder_norm
        )
        self.dropout = nn.Dropout(dropout)

    def sentences_select(self,a,q,o,topK):
        
        inner_mul_pq = torch.matmul(a,q.transpose(0,1)) # 这里用内积-矩阵乘法，除以模就是两个矩阵的二范数 128 * 1 * 1* 96 这样然后对应元素相除
        mo_pq = torch.matmul(torch.norm(a,dim=-1).unsqueeze(-1),torch.norm(q,dim=-1).unsqueeze(0)).sqrt()
        cosine_pq = inner_mul_pq/mo_pq
        inner_mul_po = torch.matmul(a,o.transpose(0,1)) # 这里用内积-矩阵乘法，除以模就是两个矩阵的二范数 128 * 1 * 1* 96 这样然后对应元素相除
        mo_po = torch.matmul(torch.norm(a,dim=-1).unsqueeze(-1),torch.norm(o,dim=-1).unsqueeze(0)).sqrt()
        cosine_po = inner_mul_po/mo_po
        Score_q = torch.div(torch.sum(torch.max(cosine_pq,dim=1)[0],dim=-1),q.size(0))#[1][0:topK]torch.max(
        Score_o = torch.div(torch.sum(torch.max(cosine_po,dim=1)[0],dim=-1),o.size(0))#[1][0:topK]torch.max(
        Rank = torch.sort(torch.sort(torch.add(Score_q,Score_o),descending=True)[1][0:topK],descending=False)[0].cpu().numpy().tolist() # 按照文章顺序组合
        for i in range(topK-len(Rank)): # 优化小技巧，在数据处理上对所有小于5的直接repeat出5个来
            Rank.append(0)
        return torch.cat([a[i] for i in Rank],dim=0).unsqueeze(0)
        # return Rank

    def forward(self, batch):
        o0_ids,o1_ids,o2_ids,o3_ids,o4_ids = batch[0],batch[1],batch[2],batch[3],batch[4]
        q_ids,a_len,a_ids = batch[5],batch[6],batch[7]

        sentences_emb_o = []
        # question and option interaction by Bert
        o0_emb = self.word_embedding(input_ids=o0_ids)
        o1_emb = self.word_embedding(input_ids=o1_ids)
        o2_emb = self.word_embedding(input_ids=o2_ids)
        o3_emb = self.word_embedding(input_ids=o3_ids)
        o4_emb = self.word_embedding(input_ids=o4_ids)
        q_emb = self.word_embedding(input_ids=q_ids)
        # paragraph selection
        for batch_id,batch_sentences in enumerate(a_ids):
            sentences_emb_o.append(self.sentences_select(self.word_embedding(input_ids=batch_sentences.transpose(0,1)),
                q_emb[batch_id],torch.cat([o0_emb[batch_id],o1_emb[batch_id],o2_emb[batch_id],o3_emb[batch_id],o4_emb[batch_id]],dim=0),3))
            # sentences_emb_o1.append(self.sentences_select(self.word_embedding(input_ids=batch_sentences.transpose(0,1)),q_emb[batch_id],o1_emb[batch_id],3))
            # sentences_emb_o2.append(self.sentences_select(self.word_embedding(input_ids=batch_sentences.transpose(0,1)),q_emb[batch_id],o2_emb[batch_id],3))
            # sentences_emb_o3.append(self.sentences_select(self.word_embedding(input_ids=batch_sentences.transpose(0,1)),q_emb[batch_id],o3_emb[batch_id],3))
            # sentences_emb_o4.append(self.sentences_select(self.word_embedding(input_ids=batch_sentences.transpose(0,1)),q_emb[batch_id],o4_emb[batch_id],3))
        sentences_o = torch.cat(sentences_emb_o,dim=0)
        # options interaction
        o0_emb_new = self.oo_attention(o0_emb,[o1_emb,o2_emb,o3_emb,o4_emb])
        o1_emb_new = self.oo_attention(o1_emb,[o0_emb,o2_emb,o3_emb,o4_emb])
        o2_emb_new = self.oo_attention(o2_emb,[o0_emb,o1_emb,o3_emb,o4_emb])
        o3_emb_new = self.oo_attention(o3_emb,[o0_emb,o1_emb,o2_emb,o4_emb])
        o4_emb_new = self.oo_attention(o4_emb,[o0_emb,o1_emb,o2_emb,o3_emb])
        # option and article interaction
        qa0 = torch.mean(self.encoder(torch.cat([q_emb,sentences_o],dim=1).transpose(0,1)),dim=0,keepdim=True)
        ao0 = torch.mean(self.encoder(torch.cat([sentences_o,o0_emb_new],dim=1).transpose(0,1)),dim=0,keepdim=True)
        qo0 = torch.mean(self.encoder(torch.cat([q_emb,o0_emb_new],dim=1).transpose(0,1)),dim=0,keepdim=True)

        qa1 = torch.mean(self.encoder(torch.cat([q_emb,sentences_o],dim=1).transpose(0,1)),dim=0,keepdim=True)
        ao1 = torch.mean(self.encoder(torch.cat([sentences_o,o1_emb_new],dim=1).transpose(0,1)),dim=0,keepdim=True)
        qo1 = torch.mean(self.encoder(torch.cat([q_emb,o1_emb_new],dim=1).transpose(0,1)),dim=0,keepdim=True)

        qa2 = torch.mean(self.encoder(torch.cat([q_emb,sentences_o],dim=1).transpose(0,1)),dim=0,keepdim=True)
        ao2 = torch.mean(self.encoder(torch.cat([sentences_o,o2_emb_new],dim=1).transpose(0,1)),dim=0,keepdim=True)
        qo2 = torch.mean(self.encoder(torch.cat([q_emb,o2_emb_new],dim=1).transpose(0,1)),dim=0,keepdim=True)

        qa3 = torch.mean(self.encoder(torch.cat([q_emb,sentences_o],dim=1).transpose(0,1)),dim=0,keepdim=True)
        ao3 = torch.mean(self.encoder(torch.cat([sentences_o,o3_emb_new],dim=1).transpose(0,1)),dim=0,keepdim=True)
        qo3 = torch.mean(self.encoder(torch.cat([q_emb,o3_emb_new],dim=1).transpose(0,1)),dim=0,keepdim=True)

        qa4 = torch.mean(self.encoder(torch.cat([q_emb,sentences_o],dim=1).transpose(0,1)),dim=0,keepdim=True)
        ao4 = torch.mean(self.encoder(torch.cat([sentences_o,o4_emb_new],dim=1).transpose(0,1)),dim=0,keepdim=True)
        qo4 = torch.mean(self.encoder(torch.cat([q_emb,o4_emb_new],dim=1).transpose(0,1)),dim=0,keepdim=True)

        option0 = torch.cat([qa0,ao0,qo0],dim=-1)
        option1 = torch.cat([qa1,ao1,qo1],dim=-1)
        option2 = torch.cat([qa2,ao2,qo2],dim=-1)
        option3 = torch.cat([qa3,ao3,qo3],dim=-1)
        option4 = torch.cat([qa4,ao4,qo4],dim=-1)

        all_option = torch.cat((option0,option1,option2,option3,option4), dim=0).transpose(0,1)

        logit =self.final_liear(all_option).squeeze(-1)
        logit = self.softmax(logit)

        return logit



        
        








            









        












