import os

import torch
from transformers import BertConfig, BertModel, BertPreTrainedModel

class BertEmb(BertPreTrainedModel):
    
    def __init__(self,config):
        super(BertEmb,self).__init__(config):
        self.bert = BertModel(config)

    def forward(self,input_ids,attention_mask=None,token_type_ids=None, position_ids=None, head_mask=None):
        device = input_ids.device
        sentence_embedding = self.bert(input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    position=position_ids,
                                    head_mask=head_mask)
        return sentence_embedding[0]