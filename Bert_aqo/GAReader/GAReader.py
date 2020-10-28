# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import XLNetModel

from Bert_aqo.Models.UnOrderedLSTM import LSTM
from Bert_aqo.Models.Linear import Linear
from Bert_aqo.Models.MLPAttention import MLPAttention
from Bert_aqo.Models.BertEmbedding import BertEmb,Qa_attention_Emb,OO_interaction


# def BiMatching(article,question): # ariticle,question [batch * hidden]


class GAReader(nn.Module):
    """
    Some difference between our GAReader and the original GAReader
    1. The query GRU is shared across hops.
    2. Dropout is applied to all hops (including the initial hop).
    3. Gated-attention is applied at the final layer as well.
    4. No character-level embeddings are used.
    """

    # def __init__(self, embedding_dim, output_dim, hidden_size, rnn_num_layers, ga_layers, bidirectional, dropout, word_emb):
    def __init__(self, embedding_dim, output_dim, hidden_size, rnn_num_layers, ga_layers, bidirectional, dropout, bert_config):
        super(GAReader, self).__init__()

        # self.word_embedding = nn.Embedding.from_pretrained(word_emb, freeze=False)
        self.word_embedding = BertEmb(bert_config)

        self.BiMatching = Qa_attention_Emb(embedding_dim)
        self.oo_attention = OO_interaction(embedding_dim,output_dim)
        self.final_liear = nn.Linear(embedding_dim * 3, 1)
        self.softmax = nn.Softmax(dim = 1)
        # self.batch_norm = nn.BatchNorm1d() 
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

    def forward(self, batch):
        o0_ids,o1_ids,o2_ids,o3_ids,o4_ids = batch[0],batch[1],batch[2],batch[3],batch[4]
        q_ids,a_len,a_ids = batch[5],batch[6],batch[7]
        # sentences_emb_o0 = []
        # sentences_emb_o1 = []
        # sentences_emb_o2 = []
        # sentences_emb_o3 = []
        # sentences_emb_o4 = []
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
                q_emb[batch_id],torch.cat([o0_emb[batch_id],o1_emb[batch_id],o2_emb[batch_id],o3_emb[batch_id],o4_emb[batch_id]],dim=0),5))
            # sentences_emb_o1.append(self.sentences_select(self.word_embedding(input_ids=batch_sentences.transpose(0,1)),q_emb[batch_id],o1_emb[batch_id],3))
            # sentences_emb_o2.append(self.sentences_select(self.word_embedding(input_ids=batch_sentences.transpose(0,1)),q_emb[batch_id],o2_emb[batch_id],3))
            # sentences_emb_o3.append(self.sentences_select(self.word_embedding(input_ids=batch_sentences.transpose(0,1)),q_emb[batch_id],o3_emb[batch_id],3))
            # sentences_emb_o4.append(self.sentences_select(self.word_embedding(input_ids=batch_sentences.transpose(0,1)),q_emb[batch_id],o4_emb[batch_id],3))
        sentences_o = torch.cat(sentences_emb_o,dim=0)
            # sentences_o1 = torch.cat(sentences_emb_o1,dim=0)
            # sentences_o2 = torch.cat(sentences_emb_o2,dim=0)
            # sentences_o3 = torch.cat(sentences_emb_o3,dim=0)
            # sentences_o4 = torch.cat(sentences_emb_o4,dim=0)
            # article_emb = self.word_embedding(input_ids=a_ids,max_l=a_len)
        # options interaction
        o0_emb_new = self.oo_attention(o0_emb,[o1_emb,o2_emb,o3_emb,o4_emb])
        o1_emb_new = self.oo_attention(o1_emb,[o0_emb,o2_emb,o3_emb,o4_emb])
        o2_emb_new = self.oo_attention(o2_emb,[o0_emb,o1_emb,o3_emb,o4_emb])
        o3_emb_new = self.oo_attention(o3_emb,[o0_emb,o1_emb,o2_emb,o4_emb])
        o4_emb_new = self.oo_attention(o4_emb,[o0_emb,o1_emb,o2_emb,o3_emb])
        # option and article interaction
        ao0_emb = self.dropout(self.BiMatching(sentences_o,o0_emb_new))
        qa0_emb = self.dropout(self.BiMatching(sentences_o,q_emb))
        qo0_emb = self.dropout(self.BiMatching(o0_emb_new,q_emb))
        ao1_emb = self.dropout(self.BiMatching(sentences_o,o1_emb_new))
        qa1_emb = self.dropout(self.BiMatching(sentences_o,q_emb))
        qo1_emb = self.dropout(self.BiMatching(o1_emb_new,q_emb))
        ao2_emb = self.dropout(self.BiMatching(sentences_o,o2_emb_new))
        qa2_emb = self.dropout(self.BiMatching(sentences_o,q_emb))
        qo2_emb = self.dropout(self.BiMatching(o2_emb_new,q_emb))
        ao3_emb = self.dropout(self.BiMatching(sentences_o,o3_emb_new))
        qa3_emb = self.dropout(self.BiMatching(sentences_o,q_emb))
        qo3_emb = self.dropout(self.BiMatching(o3_emb_new,q_emb))
        ao4_emb = self.dropout(self.BiMatching(sentences_o,o4_emb_new))
        qa4_emb = self.dropout(self.BiMatching(sentences_o,q_emb))
        qo4_emb = self.dropout(self.BiMatching(o4_emb_new,q_emb))
        option0 = torch.cat([ao0_emb,qa0_emb,qo0_emb],dim=-1)
        option1 = torch.cat([ao1_emb,qa1_emb,qo1_emb],dim=-1)
        option2 = torch.cat([ao2_emb,qa2_emb,qo2_emb],dim=-1)
        option3 = torch.cat([ao3_emb,qa3_emb,qo3_emb],dim=-1)
        option4 = torch.cat([ao4_emb,qa4_emb,qo4_emb],dim=-1)

        all_option = torch.cat((option0.unsqueeze(1),option1.unsqueeze(1),option2.unsqueeze(1),option3.unsqueeze(1),option4.unsqueeze(1)), dim=1)

        logit =self.final_liear(all_option).squeeze(-1)
        logit = self.softmax(logit)

        return logit



        
        








            









        












