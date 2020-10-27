# -*- coding: utf-8 -*-
import os

import jsonlines
import torch
from torch.utils.data import TensorDataset
from torchtext import data, datasets, vocab
from tqdm import tqdm


# test 目前先设置为dev
def load_data(path, device, tokenizer, cache_dir,max_query,max_article):
    files = os.listdir(path)
    for each_file in files:
        all_qa0_ids = []
        all_qa1_ids = []
        all_qa2_ids = []
        all_qa3_ids = []
        all_qa4_ids = []
        all_len = []
        all_label = []
        with open(path+'/'+each_file, mode='r') as f:
            reader = jsonlines.Reader(f)
            for instance in tqdm(reader):
                # 分词
                each_question0_tokens = tokenizer.tokenize(instance['question'].replace('@placeholder',instance['option_0']))
                each_question1_tokens = tokenizer.tokenize(instance['question'].replace('@placeholder',instance['option_1']))
                each_question2_tokens = tokenizer.tokenize(instance['question'].replace('@placeholder',instance['option_2']))
                each_question3_tokens = tokenizer.tokenize(instance['question'].replace('@placeholder',instance['option_3']))
                each_question4_tokens = tokenizer.tokenize(instance['question'].replace('@placeholder',instance['option_4']))
                each_article_tokens = tokenizer.tokenize(instance['article'])
                each_article_ids = tokenizer.convert_tokens_to_ids(each_article_tokens)
                each_question0_ids = tokenizer.convert_tokens_to_ids(each_question0_tokens)
                each_question1_ids = tokenizer.convert_tokens_to_ids(each_question1_tokens)
                each_question2_ids = tokenizer.convert_tokens_to_ids(each_question2_tokens)
                each_question3_ids = tokenizer.convert_tokens_to_ids(each_question3_tokens)
                each_question4_ids = tokenizer.convert_tokens_to_ids(each_question4_tokens)

                each_qa0_input = tokenizer.build_inputs_with_special_tokens(each_article_ids,each_question0_ids)
                each_qa1_input = tokenizer.build_inputs_with_special_tokens(each_article_ids,each_question1_ids)
                each_qa2_input = tokenizer.build_inputs_with_special_tokens(each_article_ids,each_question2_ids)
                each_qa3_input = tokenizer.build_inputs_with_special_tokens(each_article_ids,each_question3_ids)
                each_qa4_input = tokenizer.build_inputs_with_special_tokens(each_article_ids,each_question4_ids)
                each_len = max(len(each_qa0_input),len(each_qa1_input),len(each_qa2_input),len(each_qa3_input),len(each_qa4_input))
                
                # 规范化 问题长度
                if(max_article+max_query >= len(each_qa0_input)):
                    for i in range(max_article+max_query - len(each_qa0_input)):
                        each_qa0_input.append(0)
                else:
                    each_qa0_input = each_qa0_input[len(each_qa0_input)-max_article-max_query:len(each_qa0_input)]


                if(max_article+max_query >= len(each_qa1_input)):
                    for i in range(max_article+max_query - len(each_qa1_input)):
                        each_qa1_input.append(0)
                else:
                    each_qa1_input = each_qa1_input[len(each_qa1_input)-max_article-max_query:len(each_qa1_input)]


                if(max_article+max_query >= len(each_qa2_input)):
                    for i in range(max_article+max_query - len(each_qa2_input)):
                        each_qa2_input.append(0)
                else:
                    each_qa2_input = each_qa2_input[len(each_qa2_input)-max_article-max_query:len(each_qa2_input)]


                if(max_article+max_query >= len(each_qa3_input)):
                    for i in range(max_article+max_query - len(each_qa3_input)):
                        each_qa3_input.append(0)
                else:
                    each_qa3_input = each_qa3_input[len(each_qa3_input)-max_article-max_query:len(each_qa3_input)]


                if(max_article+max_query >= len(each_qa4_input)):
                    for i in range(max_article+max_query - len(each_qa4_input)):
                        each_qa4_input.append(0)
                else:
                    each_qa4_input = each_qa4_input[len(each_qa4_input)-max_article-max_query:len(each_qa4_input)]
                
                    
                assert len(each_qa0_input) == max_article+max_query
                assert len(each_qa1_input) == max_article+max_query
                assert len(each_qa2_input) == max_article+max_query
                assert len(each_qa3_input) == max_article+max_query
                assert len(each_qa4_input) == max_article+max_query

                # 集合id
                all_qa0_ids.append(torch.tensor(each_qa0_input,dtype=torch.long))
                all_qa1_ids.append(torch.tensor(each_qa1_input,dtype=torch.long))
                all_qa2_ids.append(torch.tensor(each_qa2_input,dtype=torch.long))
                all_qa3_ids.append(torch.tensor(each_qa3_input,dtype=torch.long))
                all_qa4_ids.append(torch.tensor(each_qa4_input,dtype=torch.long))
                all_len.append(min(each_len,max_article+max_query))
                all_label.append(instance['label'])
        if('train' in each_file):
            train_dataset = TensorDataset(all_qa0_ids,all_qa1_ids,all_qa2_ids,all_qa3_ids,all_qa4_ids,all_len,all_label)
        else:
            dev_dataset = TensorDataset(all_qa0_ids,all_qa1_ids,all_qa2_ids,all_qa3_ids,all_qa4_ids,all_len,all_label)
            test_dataset = TensorDataset(all_qa0_ids,all_qa1_ids,all_qa2_ids,all_qa3_ids,all_qa4_ids,all_len)

    return train_dataset,dev_dataset,test_dataset

