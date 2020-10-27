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
        all_q0_ids,all_q1_ids,all_q2_ids,all_q3_ids,all_q4_ids,all_a_ids = [],[],[],[],[],[]
        all_q_len,all_a_len = [],[]
        # all_option0_ids,all_option1_ids,all_option2_ids,all_option3_ids,all_option4_ids = [],[],[],[],[]
        all_label = []
        with open(path+'/'+each_file, mode='r') as f:
            reader = jsonlines.Reader(f)
            for instance in tqdm(reader):
                # 分词
                # input_ids = torch.tensor(tokenizer.encode(instance['question'])) 
                
                each_query0_tokens = tokenizer.tokenize(instance['question'].replace('@placeholder',instance['option_0']))
                each_query1_tokens = tokenizer.tokenize(instance['question'].replace('@placeholder',instance['option_1']))
                each_query2_tokens = tokenizer.tokenize(instance['question'].replace('@placeholder',instance['option_2']))
                each_query3_tokens = tokenizer.tokenize(instance['question'].replace('@placeholder',instance['option_3']))
                each_query4_tokens = tokenizer.tokenize(instance['question'].replace('@placeholder',instance['option_4']))

                each_article_tokens = tokenizer.tokenize(instance['article'])
                # 构建id
                # option0_id = tokenizer.convert_tokens_to_ids(instance['option_0'])
                # option1_id = tokenizer.convert_tokens_to_ids(instance['option_1'])
                # option2_id = tokenizer.convert_tokens_to_ids(instance['option_2'])
                # option3_id = tokenizer.convert_tokens_to_ids(instance['option_3'])
                # option4_id = tokenizer.convert_tokens_to_ids(instance['option_4'])
                each_query0_ids =tokenizer.convert_tokens_to_ids(each_query0_tokens)
                each_query1_ids =tokenizer.convert_tokens_to_ids(each_query1_tokens)
                each_query2_ids =tokenizer.convert_tokens_to_ids(each_query2_tokens)
                each_query3_ids =tokenizer.convert_tokens_to_ids(each_query3_tokens)
                each_query4_ids =tokenizer.convert_tokens_to_ids(each_query4_tokens)
                each_article_ids  =tokenizer.convert_tokens_to_ids(each_article_tokens)
                # 构建 input id
                # option0_input = tokenizer.build_inputs_with_special_tokens([option0_id])
                # option1_input = tokenizer.build_inputs_with_special_tokens([option1_id])
                # option2_input = tokenizer.build_inputs_with_special_tokens([option2_id])
                # option3_input = tokenizer.build_inputs_with_special_tokens([option3_id])
                # option4_input = tokenizer.build_inputs_with_special_tokens([option4_id])
                each_q0_input_ids = tokenizer.build_inputs_with_special_tokens(each_query0_ids)
                each_q1_input_ids = tokenizer.build_inputs_with_special_tokens(each_query1_ids)
                each_q2_input_ids = tokenizer.build_inputs_with_special_tokens(each_query2_ids)
                each_q3_input_ids = tokenizer.build_inputs_with_special_tokens(each_query3_ids)
                each_q4_input_ids = tokenizer.build_inputs_with_special_tokens(each_query4_ids)
                each_a_input_ids = tokenizer.build_inputs_with_special_tokens(each_article_ids)

                each_a_len = len(each_a_input_ids)
                # each_q_len = len(each_q_input_ids)
                # 构建 segment
                # each_q_segment = tokenizer.create_token_type_ids_from_sequences(each_query_ids)
                # each_a_segment = tokenizer.create_token_type_ids_from_sequences(each_article_ids)
                # option0_segment = tokenizer.create_token_type_ids_from_sequences([option0_id])
                # option1_segment = tokenizer.create_token_type_ids_from_sequences([option1_id])
                # option2_segment = tokenizer.create_token_type_ids_from_sequences([option2_id])
                # option3_segment = tokenizer.create_token_type_ids_from_sequences([option3_id])
                # option4_segment = tokenizer.create_token_type_ids_from_sequences([option4_id])
                # 构建 mask 
                # each_q_mask = [1] * len(each_q_input_ids)
                # each_a_mask = [1] * len(each_a_input_ids)
                # option0_mask = [1] * len([option0_input])
                # option1_mask = [1] * len([option1_input])
                # option2_mask = [1] * len([option2_input])
                # option3_mask = [1] * len([option3_input])
                # option4_mask = [1] * len([option4_input])
                # 规范化 问题长度
                for i in range(max_query - len(each_q0_input_ids)):
                    each_q0_input_ids.append(0)
                for i in range(max_query - len(each_q1_input_ids)):
                    each_q1_input_ids.append(0)
                for i in range(max_query - len(each_q2_input_ids)):
                    each_q2_input_ids.append(0)
                for i in range(max_query - len(each_q3_input_ids)):
                    each_q3_input_ids.append(0)
                for i in range(max_query - len(each_q4_input_ids)):
                    each_q4_input_ids.append(0)
                assert len(each_q0_input_ids) == max_query
                assert len(each_q1_input_ids) == max_query
                assert len(each_q2_input_ids) == max_query
                assert len(each_q3_input_ids) == max_query
                assert len(each_q4_input_ids) == max_query
                # 规范化 文章长度
                for i in range(max_article - len(each_a_input_ids)):
                    each_a_input_ids.append(0)
                assert len(each_a_input_ids) == max_article
                # 集合id
                all_q0_ids.append(torch.tensor(each_q0_input_ids,dtype=torch.long))
                all_q1_ids.append(torch.tensor(each_q1_input_ids,dtype=torch.long))
                all_q2_ids.append(torch.tensor(each_q2_input_ids,dtype=torch.long))
                all_q3_ids.append(torch.tensor(each_q3_input_ids,dtype=torch.long))
                all_q4_ids.append(torch.tensor(each_q4_input_ids,dtype=torch.long))
                all_a_ids.append(torch.tensor(each_a_input_ids,dtype=torch.long))
                # all_option0_ids.append(torch.tensor(option0_input,dtype=torch.long))
                # all_option1_ids.append(torch.tensor(option1_input,dtype=torch.long))
                # all_option2_ids.append(torch.tensor(option2_input,dtype=torch.long))
                # all_option3_ids.append(torch.tensor(option3_input,dtype=torch.long))
                # all_option4_ids.append(torch.tensor(option4_input,dtype=torch.long))
                all_label.append(instance['label'])
                all_a_len.append(each_a_len)
                # all_q_len.append(each_q_len)
        if('train' in each_file):
            # train_dataset = TensorDataset(all_q_ids,all_q_seg,all_q_mask,all_a_ids,all_a_seg,all_a_mask,
            #         all_option0_ids,all_option0_seg,all_option0_mask,
            #         all_option1_ids,all_option1_seg,all_option1_mask,
            #         all_option2_ids,all_option2_seg,all_option2_mask,
            #         all_option3_ids,all_option3_seg,all_option3_mask,
            #         all_option4_ids,all_option4_seg,all_option4_mask,
            #         all_label)
            train_dataset = TensorDataset(all_q0_ids,all_q1_ids,all_q2_ids,all_q3_ids,all_q4_ids,all_a_ids,all_a_len,all_label)
        else:
            # dev_dataset = TensorDataset(all_q_ids,all_q_seg,all_q_mask,all_a_ids,all_a_seg,all_a_mask,
            #         all_option0_ids,all_option0_seg,all_option0_mask,
            #         all_option1_ids,all_option1_seg,all_option1_mask,
            #         all_option2_ids,all_option2_seg,all_option2_mask,
            #         all_option3_ids,all_option3_seg,all_option3_mask,
            #         all_option4_ids,all_option4_seg,all_option4_mask,
            #         all_label)
            dev_dataset = TensorDataset(all_q0_ids,all_q1_ids,all_q2_ids,all_q3_ids,all_q4_ids,all_a_ids,all_a_len,all_label)
            test_dataset = TensorDataset(all_q0_ids,all_q1_ids,all_q2_ids,all_q3_ids,all_q4_ids,all_a_ids,all_a_len)


    # # word_vectors = vocab.Vectors(word_embed_file, cache=cache_dir)

    # train, dev, test = data.TabularDataset.splits(
    #     path=path, train='Task_1_train.jsonl', validation='Task_1_dev.jsonl',test='Task_1_dev.jsonl',
    #      format='json', fields=fields)
    #     # test='test.jsonl', format='json', fields=fields) 'Task_1_dev.jsonl'
    
    # print("the size of train: {}, dev:{}, test:{}".format(
    #     len(train.examples), len(dev.examples), len(test.examples)))
    
    # word_field.build_vocab(train, dev, test, max_size=50000)
    
    # label_field.build_vocab(train, dev, test)
    
    # train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    #     (train, dev, test), batch_sizes=(train_batch_size, dev_batch_size, test_batch_size), sort_key=lambda x: len(x.article), device=device, shuffle=True)

    return train_dataset,dev_dataset,test_dataset

