# -*- coding: utf-8 -*-
import os

import jsonlines
import torch
from torch.utils.data import TensorDataset
from torchtext import data, datasets, vocab
from tqdm import tqdm


# test 目前先设置为dev
def load_data(path, device, tokenizer, cache_dir,max_query,max_sentence,max_option):
    files = os.listdir(path)
    for each_file in files:
        all_o0_ids,all_o1_ids,all_o2_ids,all_o3_ids,all_o4_ids,all_a_ids,all_q_ids = [],[],[],[],[],[],[]
        all_a_len = []
        # all_option0_ids,all_option1_ids,all_option2_ids,all_option3_ids,all_option4_ids = [],[],[],[],[]
        all_label = []
        with open(path+'/'+each_file, mode='r') as f:
            reader = jsonlines.Reader(f)
            cnt = 0
            for instance in tqdm(reader):
                # 分词
                cnt += 1
                # input_ids = torch.tensor(tokenizer.encode(instance['question'])) 
                each_article_input_ids = []
                if('.' not in instance['article']):
                    sentences = instance['article'].split('.')
                else:
                    sentences = instance['article'].split('.')[0:-1]
                for each_sentence in sentences:
                    sentence_tokens = tokenizer.tokenize(each_sentence)
                    each_article_ids  =tokenizer.convert_tokens_to_ids(sentence_tokens)
                    each_a_input_ids = tokenizer.build_inputs_with_special_tokens(each_article_ids)
                     # 规范化 文章句子长度
                    if(len(each_a_input_ids)>max_sentence):
                        each_a_input_ids = each_a_input_ids[0:max_sentence]
                    for i in range(max_sentence - len(each_a_input_ids)):
                        each_a_input_ids.append(0)
                    assert len(each_a_input_ids) == max_sentence
                    each_article_input_ids.append(torch.tensor(each_a_input_ids,dtype=torch.long).unsqueeze(-1))

                each_query_tokens = tokenizer.tokenize(instance['question'])

                each_option0_tokens = tokenizer.tokenize(instance['option_0'])
                each_option1_tokens = tokenizer.tokenize(instance['option_1'])
                each_option2_tokens = tokenizer.tokenize(instance['option_2'])
                each_option3_tokens = tokenizer.tokenize(instance['option_3'])
                each_option4_tokens = tokenizer.tokenize(instance['option_4'])

                # each_article_tokens = tokenizer.tokenize(instance['article'])
                # 构建id
             
                each_query_ids =tokenizer.convert_tokens_to_ids(each_query_tokens)
                each_option0_ids =tokenizer.convert_tokens_to_ids(each_option0_tokens)
                each_option1_ids =tokenizer.convert_tokens_to_ids(each_option1_tokens)
                each_option2_ids =tokenizer.convert_tokens_to_ids(each_option2_tokens)
                each_option3_ids =tokenizer.convert_tokens_to_ids(each_option3_tokens)
                each_option4_ids =tokenizer.convert_tokens_to_ids(each_option4_tokens)
                
                # 构建 input id
                
                each_q_input_ids = tokenizer.build_inputs_with_special_tokens(each_query_ids)
                each_option0_input_ids = tokenizer.build_inputs_with_special_tokens(each_option0_ids)
                each_option1_input_ids = tokenizer.build_inputs_with_special_tokens(each_option1_ids)
                each_option2_input_ids = tokenizer.build_inputs_with_special_tokens(each_option2_ids)
                each_option3_input_ids = tokenizer.build_inputs_with_special_tokens(each_option3_ids)
                each_option4_input_ids = tokenizer.build_inputs_with_special_tokens(each_option4_ids)
                # each_a_input_ids = tokenizer.build_inputs_with_special_tokens(each_article_ids)

                # each_a_len = len(each_a_input_ids)
                # each_q_len = len(each_q_input_ids)
             
                # 规范化 问题长度
                for i in range(max_option - len(each_option0_input_ids)):
                    each_option0_input_ids.append(0)
                for i in range(max_option - len(each_option1_input_ids)):
                    each_option1_input_ids.append(0)
                for i in range(max_option - len(each_option2_input_ids)):
                    each_option2_input_ids.append(0)
                for i in range(max_option - len(each_option3_input_ids)):
                    each_option3_input_ids.append(0)
                for i in range(max_option - len(each_option4_input_ids)):
                    each_option4_input_ids.append(0)
                assert len(each_option0_input_ids) == max_option
                assert len(each_option1_input_ids) == max_option
                assert len(each_option2_input_ids) == max_option
                assert len(each_option3_input_ids) == max_option
                assert len(each_option4_input_ids) == max_option
               
                # if(len(each_a_input_ids)>max_sentence):
                #     each_a_input_ids = each_a_input_ids[0:max_sentence]
                # for i in range(max_sentence - len(each_a_input_ids)):
                #     each_a_input_ids.append(0)
                # assert len(each_a_input_ids) == max_sentence
                for i in range(max_query - len(each_q_input_ids)):
                    each_q_input_ids.append(0)
                assert len(each_q_input_ids) == max_query
                # 集合id
                all_o0_ids.append(torch.tensor(each_option0_input_ids,dtype=torch.long))
                all_o1_ids.append(torch.tensor(each_option1_input_ids,dtype=torch.long))
                all_o2_ids.append(torch.tensor(each_option2_input_ids,dtype=torch.long))
                all_o3_ids.append(torch.tensor(each_option3_input_ids,dtype=torch.long))
                all_o4_ids.append(torch.tensor(each_option4_input_ids,dtype=torch.long))
                all_a_ids.append(torch.cat(each_article_input_ids,dim=-1))
                all_q_ids.append(torch.tensor(each_q_input_ids,dtype=torch.long))

                all_label.append(instance['label'])
                all_a_len.append(len(each_article_input_ids))
                # all_q_len.append(each_q_len)
        if('train' in each_file):

            train_dataset = TensorDataset(all_o0_ids,all_o1_ids,all_o2_ids,all_o3_ids,all_o4_ids,all_a_ids,all_q_ids,all_a_len,all_label)
        else:
            dev_dataset = TensorDataset(all_o0_ids,all_o1_ids,all_o2_ids,all_o3_ids,all_o4_ids,all_a_ids,all_q_ids,all_a_len,all_label)
            test_dataset = TensorDataset(all_o0_ids,all_o1_ids,all_o2_ids,all_o3_ids,all_o4_ids,all_a_ids,all_q_ids,all_a_len,all_label)


    return train_dataset,dev_dataset,test_dataset

