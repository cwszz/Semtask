import math
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_utils import SequenceSummary

# from transformers import XLNetConfig, XLNetModel,XLNetPreTrainedModel,XLNetForMultipleChoice

class BertEmb(BertPreTrainedModel):
    
    def __init__(self,config):
        super(BertEmb,self).__init__(config)
        self.config = config
        self.bert = BertModel(config)
        # self.windows_cat = SequenceSummary(config)

    def cut(self,input_ids,max_article):
        stride = 256
        window_size = 512
        max_len = max(0,torch.max(max_article)-window_size)
        if(max_len % stride != 0):
            step = max_len // stride + 1
        else:
            step = max_len // stride
        g = [input_ids[:,i*stride:min(input_ids.size(1),i*stride+window_size)].unsqueeze(1) for i in range(min(3,step+1))]
        tt = torch.cat(g,dim=1).reshape(-1,window_size)
        # ttt = torch.reshape(tt,[tt.size(0)*tt.size(1),-1])
        return tt
         
    def forward(self,input_ids,attention_mask=None,token_type_ids=None, position_ids=None, head_mask=None,max_l=None):
        if(max_l is None ):
            # print("nnn")
            input_id_sets = input_ids
        else:
            input_id_sets = self.cut(input_ids,max_l)
        if(input_id_sets.size(0) == 1):
            return self.bert(input_id_sets,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask)[0]
        sentence_embedding = self.bert(input_id_sets,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask)
        
        Sequence_emb = sentence_embedding[0].reshape(input_ids.size(0),-1,self.config.hidden_size)
        # if(Sequence_emb.size(1)>768):
        #     sentence_encode = sentence_embedding[1].reshape(input_ids.size(0),-1,self.config.hidden_size)
        # sentence_embedding = self.windows_cat(sentence_embedding)
        # sentence_embedding = torch.sum(sentence_embedding,dim=-1)
        # sentence_embedding = sentence_embedding.reshape(input_ids.size(0),-1)
                    # .reshape(input_ids.size(0),input_id_sets.size(1)*input_id_sets.size(0)//input_ids.size(0),-1)
        
        return Sequence_emb

class Qa_attention_Emb(nn.Module):

    def __init__(self,input_size):
        super(Qa_attention_Emb, self).__init__()
        self.hidden_size = input_size
        self.weight1_1 = Parameter(torch.Tensor(input_size,input_size))
        self.weight1_2 = Parameter(torch.Tensor(input_size,input_size))
        self.weight2_1 = Parameter(torch.Tensor(input_size,input_size))
        self.weight2_2 = Parameter(torch.Tensor(input_size,input_size))
        # self.distribution_1 = Parameter(torch.Tensor(input_size,input_size))
        # self.distribution_2 = Parameter(torch.Tensor(input_size,input_size))
        self.bias = Parameter(torch.Tensor(input_size))
        # self.normal = nn.LayerNorm(768)
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight1_1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight1_2, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2_1, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight2_2, a=math.sqrt(5))
        # init.kaiming_uniform_(self.distribution_1, a=math.sqrt(5))
        # init.kaiming_uniform_(self.distribution_2, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight1_1)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self,article_feature,query_op_feature,softmax_dim=2):
        if(article_feature.size(2)!=self.hidden_size or query_op_feature.size(2)!=self.hidden_size):
            print("Error")
        else:
            
            aq_attention = F.softmax(torch.bmm(article_feature.matmul(self.weight1_1),query_op_feature.transpose(1,2)),dim=softmax_dim)
            qa_attention = F.softmax(torch.bmm(query_op_feature.matmul(self.weight1_2),article_feature.transpose(1,2)),dim=softmax_dim)
            # gg = torch.bmm(aq_attention,query_op_feature)
            # ggg = torch.bmm(qa_attention,article_feature)
            S_a = self.relu(torch.bmm(aq_attention,query_op_feature).matmul(self.weight2_1))
            S_q = self.relu(torch.bmm(qa_attention,article_feature).matmul(self.weight2_2))
            # S_a = self.relu(self.normal(torch.bmm(aq_attention,query_op_feature).matmul(self.weight2_1)))
            # S_q = self.relu(self.normal(torch.bmm(qa_attention,article_feature).matmul(self.weight2_2)))
            S_aq = torch.max(S_a,dim=1)[0]
            S_qa = torch.max(S_q,dim=1)[0]
            # temp = torch.mm(S_aq,self.distribution_1)
            # temp2 = torch.addmm(self.bias,S_qa,self.distribution_2)
            # g =torch.sigmoid(torch.mm(S_aq,self.distribution_1)+torch.addmm(self.bias,S_qa,self.distribution_2))
            g = torch.div(torch.ones(S_qa.size()),2).to(aq_attention.device)
            return torch.mul(g,S_aq) + torch.mul((1-g),S_qa)

class QO_interaction(nn.Module):

    def __init__(self,input_size,option_num):
        super(QO_interaction, self).__init__()
        self.hidden_size = input_size
        self.weight = Parameter(torch.Tensor(input_size,input_size))
        self.weight_2 = Parameter(torch.Tensor(input_size*(option_num-1),input_size))
        self.weight_ano = Parameter(torch.Tensor(input_size,input_size))
        self.weight_ori = Parameter(torch.Tensor(input_size,input_size))
        self.bias = Parameter(torch.Tensor(128,input_size))
        # self.normal = nn.LayerNorm(768)
        self.relu = nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_2, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_ano, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_ori, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self,input_feature,addition_feature,softmax_dim=1):
        if(input_feature.size(2)!=self.hidden_size or addition_feature[0].size(2)!=self.hidden_size):
            print("Error")
        else:
            H_other = []
            first_part = F.linear(input_feature,self.weight,None)
            for each_add_feature in addition_feature:
                result = torch.bmm(first_part,each_add_feature.transpose(1,2))
                distribution = F.softmax(result,dim=softmax_dim)
                H_other.append(self.relu(torch.bmm(distribution,each_add_feature)))
                # H_other.append(self.relu(self.normal(torch.bmm(distribution,each_add_feature))))
            H = torch.cat(H_other,dim=-1)
            H_ano = H.matmul(self.weight_2)
            # g = torch.div(torch.ones(H.size(0),H.size(1),H_ano.size(-1)),2).to(device=H.device)
            g = torch.sigmoid(H_ano.matmul(self.weight_ano) + F.linear(input_feature,self.weight_ori,self.bias))
            return torch.mul(g,input_feature) + torch.mul((1-g),H_ano)

