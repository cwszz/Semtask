import os

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertPreTrainedModel

class BertEmb(BertPreTrainedModel):
    
    def __init__(self,config):
        super(BertEmb,self).__init__(config)
        self.bert = BertModel(config)

    def cut(self,input_ids,max_article,stride):
        # aa = torch.cat([input_ids[:,i:min(input_ids.size(1),i+max_article)] for i in range(0,input_ids.size(1),stride)],1)
        # g = [input_ids[i:min(input_ids.size(1),i+max_article)] for i in range(0,input_ids.size(1),stride)]
        zero_matrix = torch.zeros(input_ids.size(0),max_article,dtype=torch.long).to(input_ids.device)
        g = torch.split(input_ids,max_article,dim=1)
        tt = torch.cat([each_g.unsqueeze(-1) for each_g in g if not zero_matrix.equal(each_g)],dim=-1).transpose(1,2)
        return torch.reshape(tt,[tt.size(0)*tt.size(1),-1])
         
    def forward(self,input_ids,attention_mask=None,token_type_ids=None, position_ids=None, head_mask=None,max_l=3):
        input_id_sets = self.cut(input_ids,max_l,max_l)
        # sentence_embedding = torch.zeros(input_ids.size(0),max_l,768,dtype=torch.float).to(input_ids.device)
        # attention_mask = attention_mask.repeat(input_id_sets.size(0)//attention_mask.size(0),1)
        # token_type_ids = token_type_ids.repeat(input_id_sets.size(0)//token_type_ids.size(0),1)
        # sentence_embedding = self.bert(input_id_sets,
        #                                 attention_mask=attention_mask[:,0:512],
        #                                 token_type_ids=token_type_ids[:,0:512],
        #                                 position_ids=position_ids,
        #                                 head_mask=head_mask)[0].reshape(input_ids.size(0),input_id_sets.size(1)*input_id_sets.size(0)//input_ids.size(0),-1)
        sentence_embedding = self.bert(input_id_sets,
                        attention_mask=attention_mask.repeat(input_id_sets.size(0)//attention_mask.size(0),1)[:,0:512],
                        token_type_ids=token_type_ids.repeat(input_id_sets.size(0)//token_type_ids.size(0),1)[:,0:512],
                        position_ids=position_ids,
                        head_mask=head_mask)[0].reshape(input_ids.size(0),input_id_sets.size(1)*input_id_sets.size(0)//input_ids.size(0),-1)
        
        return sentence_embedding