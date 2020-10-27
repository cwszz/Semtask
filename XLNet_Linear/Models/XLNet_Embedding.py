import os

import torch
import torch.nn as nn
from transformers import BertConfig, XLNetModel, XLNetPreTrainedModel
from transformers.modeling_utils import SequenceSummary
# from Bert_Linear.Models.UnOrderedLSTM import LSTM
# from transformers import XLNetConfig, XLNetModel,XLNetPreTrainedModel,XLNetForMultipleChoice

class XLNet_Reader(XLNetPreTrainedModel):
    
    def __init__(self,config):
        super(XLNet_Reader,self).__init__(config)
        self.config = config
        self.xlnet = XLNetModel(config)
        self.sentence_summary = SequenceSummary(config)
        # self.dropout =nn.Dropout(0.3)
        # self.aggregation = nn.LSTM(config.hidden_size,config.hidden_size,batch_first=True,num_layers=1,bidirectional=True)
        # self.batch_norm_for_rnn = nn.BatchNorm1d(config.hidden_size)
        self.dropout = nn.Dropout(0.1)

         
    def forward(self,input_ids,attention_mask=None,token_type_ids=None, head_mask=None,max_l=3):
        # input_id_sets = self.cut(input_ids,max_l,max_l//2)
        sentence_embedding = self.xlnet(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    head_mask=head_mask)[0]
        # t = torch.split(sentence_embedding,split_size_or_sections=)
        sentence_embedding = self.sentence_summary(sentence_embedding)
                    # [1].reshape(input_ids.size(0),-1,self.config.hidden_size)
        # g = self.aggregation(sentence_embedding)[1][0].transpose(0,1).reshape(input_ids.size(0),-1)
        if(sentence_embedding.size(0) == 1):
            sentence_embedding = sentence_embedding.repeat(2,1)
        sentence_emb = self.dropout(sentence_embedding)
        
        
        return sentence_emb