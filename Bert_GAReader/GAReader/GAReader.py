# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import XLNetModel

from Bert_GAReader.Models.UnOrderedLSTM import LSTM
from Bert_GAReader.Models.Linear import Linear
from Bert_GAReader.Models.MLPAttention import MLPAttention
from Bert_GAReader.Models.BertEmbedding import BertEmb,Qa_attention_Emb,QO_interaction

def gated_attention(article, question):
    """
    Args:
        article: [batch_size, article_len , dim]
        question: [batch_size, question_len, dim]
    Returns:
        question_to_article: [batch_size, article_len, dim]
    """
    question_att = question.permute(0, 2, 1)
    # question : [batch_size * dim * question_len]

    att_matrix = torch.bmm(article, question_att)
    # att_matrix: [batch_size * article_len * question_len]

    att_weights = F.softmax(att_matrix.view(-1, att_matrix.size(-1)), dim=1).view_as(att_matrix)
    # att_weights: [batch_size, article_len, question_len]

    question_rep = torch.bmm(att_weights, question)
    # question_rep : [batch_size, article_len, dim]

    question_to_article = torch.mul(article, question_rep)
    # question_to_article: [batch_size, article_len, dim]

    return question_to_article


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
        self.qo_attention = QO_interaction(embedding_dim,output_dim)
        self.final_liear = nn.Linear(embedding_dim, 1)
        self.softmax = nn.Softmax(dim = 1)
        # self.batch_norm = nn.BatchNorm1d() 
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        qo0_ids,qo1_ids,qo2_ids,qo3_ids,qo4_ids = batch[0],batch[1],batch[2],batch[3],batch[4]
        a_ids,a_len = batch[5],batch[6]
   
        article_emb = self.word_embedding(input_ids=a_ids,max_l=a_len)
        # question and option interaction by Bert
        qo0_emb = self.word_embedding(input_ids=qo0_ids)
        qo1_emb = self.word_embedding(input_ids=qo1_ids)
        qo2_emb = self.word_embedding(input_ids=qo2_ids)
        qo3_emb = self.word_embedding(input_ids=qo3_ids)
        qo4_emb = self.word_embedding(input_ids=qo4_ids)
        # options interaction
        qo0_emb_new = self.qo_attention(qo0_emb,[qo1_emb,qo2_emb,qo3_emb,qo4_emb])
        qo1_emb_new = self.qo_attention(qo1_emb,[qo0_emb,qo2_emb,qo3_emb,qo4_emb])
        qo2_emb_new = self.qo_attention(qo2_emb,[qo0_emb,qo1_emb,qo3_emb,qo4_emb])
        qo3_emb_new = self.qo_attention(qo3_emb,[qo0_emb,qo1_emb,qo2_emb,qo4_emb])
        qo4_emb_new = self.qo_attention(qo4_emb,[qo0_emb,qo1_emb,qo2_emb,qo3_emb])
        # question_option and article interaction
        qao0_emb = self.dropout(self.BiMatching(article_emb,qo0_emb_new))
        qao1_emb = self.dropout(self.BiMatching(article_emb,qo1_emb_new))
        qao2_emb = self.dropout(self.BiMatching(article_emb,qo2_emb_new))
        qao3_emb = self.dropout(self.BiMatching(article_emb,qo3_emb_new))
        qao4_emb = self.dropout(self.BiMatching(article_emb,qo4_emb_new))

        all_option = torch.cat((qao0_emb.unsqueeze(1),qao1_emb.unsqueeze(1),qao2_emb.unsqueeze(1),qao3_emb.unsqueeze(1),qao4_emb.unsqueeze(1)), dim=1)

        logit =self.final_liear(all_option).squeeze(-1)
        logit = self.softmax(logit)

        return logit



        
        








            









        












