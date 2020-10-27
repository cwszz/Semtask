# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import XLNetModel

from XLNet_Linear.Models.UnOrderedLSTM import LSTM
from XLNet_Linear.Models.Linear import Linear
# from XLNet_Linear.Models.MLPAttention import MLPAttention
from XLNet_Linear.Models.XLNet_Embedding import XLNet_Reader

# def gated_attention(article, question):
#     """
#     Args:
#         article: [batch_size, article_len , dim]
#         question: [batch_size, question_len, dim]
#     Returns:
#         question_to_article: [batch_size, article_len, dim]
#     """
#     question_att = question.permute(0, 2, 1)
#     # question : [batch_size * dim * question_len]

#     att_matrix = torch.bmm(article, question_att)
#     # att_matrix: [batch_size * article_len * question_len]

#     att_weights = F.softmax(att_matrix.view(-1, att_matrix.size(-1)), dim=1).view_as(att_matrix)
#     # att_weights: [batch_size, article_len, question_len]

#     question_rep = torch.bmm(att_weights, question)
#     # question_rep : [batch_size, article_len, dim]

#     question_to_article = torch.mul(article, question_rep)
#     # question_to_article: [batch_size, article_len, dim]

#     return question_to_article



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
        self.word_embedding = XLNet_Reader(bert_config)

        self.final_liear = nn.Linear(embedding_dim , 1)
        # self.dropout = nn.Dropout(0.2)
        self.batch_norm_for_linear = nn.BatchNorm1d(embedding_dim) 
        self.softmax = nn.Softmax(dim =1)

    def forward(self, batch):

        # option0,option1,option2,option3,option4 = batch[0],batch[1],batch[2],batch[3],batch[4]
        option0_ids,option1_ids,option2_ids,option3_ids,option4_ids = batch[0],batch[1],batch[2],batch[3],batch[4]
       

        # article_emb = self.dropout(self.word_embedding(input_ids=a_ids,attention_mask=a_mask,token_type_ids=a_seg,max_l=512))
        # option0_emb = self.dropout(self.word_embedding(input_ids=option0_ids))
        # option1_emb = self.dropout(self.word_embedding(input_ids=option1_ids))
        # option2_emb = self.dropout(self.word_embedding(input_ids=option2_ids))
        # option3_emb = self.dropout(self.word_embedding(input_ids=option3_ids))
        # option4_emb = self.dropout(self.word_embedding(input_ids=option4_ids))

        # option0_emb = self.word_embedding(input_ids=option0_ids)
        # option1_emb = self.word_embedding(input_ids=option1_ids)
        # option2_emb = self.word_embedding(input_ids=option2_ids)
        # option3_emb = self.word_embedding(input_ids=option3_ids)
        # option4_emb = self.word_embedding(input_ids=option4_ids)
        option_ids = torch.cat([option0_ids.unsqueeze(1),option1_ids.unsqueeze(1),option2_ids.unsqueeze(1),
                    option3_ids.unsqueeze(1),option4_ids.unsqueeze(1)],dim=1).reshape(-1,option0_ids.size(-1))
        option_emb = self.word_embedding(input_ids=option_ids)
        # assert option_ids[0].equal(option0_ids[0])
        # assert option_ids[1].equal(option1_ids[0])
        all_infomation = self.batch_norm_for_linear(option_emb)
        # option_emb = torch.cat([option0_ids,option1_ids,option2_ids,option3_ids,option4_ids],dim=1)


        
        # all_infomation = torch.cat((option0_emb,option1_emb,option2_emb,option3_emb,option4_emb), dim=1)

        # all_infomation = self.batch_norm_for_linear(torch.cat((option0_emb,option1_emb,option2_emb,option3_emb,option4_emb), dim=1))

        logit = self.softmax(self.final_liear(all_infomation).reshape(option0_ids.size(0),-1))

        return logit



        
        








            









        












