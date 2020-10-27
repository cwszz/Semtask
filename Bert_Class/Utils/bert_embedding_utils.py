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
        # all_qa0_ids = []
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
                # 构建id
                # q_option0_id = tokenizer.convert_tokens_to_ids(instance['option_0'])
                # q_option1_id = tokenizer.convert_tokens_to_ids(instance['option_1'])
                # option2_id = tokenizer.convert_tokens_to_ids(instance['option_2'])
                # option3_id = tokenizer.convert_tokens_to_ids(instance['option_3'])
                # option4_id = tokenizer.convert_tokens_to_ids(instance['option_4'])
                # each_query_ids =tokenizer.convert_tokens_to_ids(each_question_tokens)
                # each_article_ids  =tokenizer.convert_tokens_to_ids(each_article_tokens)
                # 构建 input id
                # option0_input = tokenizer.build_inputs_with_special_tokens([option0_id])
                # option1_input = tokenizer.build_inputs_with_special_tokens([option1_id])
                # option2_input = tokenizer.build_inputs_with_special_tokens([option2_id])
                # option3_input = tokenizer.build_inputs_with_special_tokens([option3_id])
                # option4_input = tokenizer.build_inputs_with_special_tokens([option4_id])
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
                #     each_q_segment.append(0)
                #     each_q_mask.append(0)
                # assert len(each_q_input_ids) == max_query
                # assert len(each_q_mask) == max_query
                # assert len(each_q_segment) == max_query
                # # 规范化 文章长度
                # for i in range(max_article - len(each_a_input_ids)):
                #     each_a_input_ids.append(0)
                #     each_a_segment.append(0)
                #     each_a_mask.append(0)
                # assert len(each_a_input_ids) == max_article
                # assert len(each_a_mask) == max_article
                # assert len(each_a_segment) == max_article

                # 集合id
                all_qa0_ids.append(torch.tensor(each_qa0_input,dtype=torch.long))
                all_qa1_ids.append(torch.tensor(each_qa1_input,dtype=torch.long))
                all_qa2_ids.append(torch.tensor(each_qa2_input,dtype=torch.long))
                all_qa3_ids.append(torch.tensor(each_qa3_input,dtype=torch.long))
                all_qa4_ids.append(torch.tensor(each_qa4_input,dtype=torch.long))
                all_len.append(min(each_len,max_article+max_query))
                # all_a_ids.append(torch.tensor(each_a_input,dtype=torch.long))
                # all_option0_ids.append(torch.tensor(option0_input,dtype=torch.long))
                # all_option1_ids.append(torch.tensor(option1_input,dtype=torch.long))
                # all_option2_ids.append(torch.tensor(option2_input,dtype=torch.long))
                # all_option3_ids.append(torch.tensor(option3_input,dtype=torch.long))
                # all_option4_ids.append(torch.tensor(option4_input,dtype=torch.long))
                # 集合seg
                # all_q_seg.append(torch.tensor(each_q_segment,dtype=torch.long))
                # all_a_seg.append(torch.tensor(each_a_segment,dtype=torch.long))
                # all_option0_seg.append(torch.tensor(option0_segment,dtype=torch.long))
                # all_option1_seg.append(torch.tensor(option1_segment,dtype=torch.long))
                # all_option2_seg.append(torch.tensor(option2_segment,dtype=torch.long))
                # all_option3_seg.append(torch.tensor(option3_segment,dtype=torch.long))
                # all_option4_seg.append(torch.tensor(option4_segment,dtype=torch.long))
                # 集合mask
                # all_a_mask.append(torch.tensor(each_a_mask,dtype=torch.long))
                # all_q_mask.append(torch.tensor(each_q_mask,dtype=torch.long))
                # all_option0_mask.append(torch.tensor(option0_mask,dtype=torch.long))
                # all_option1_mask.append(torch.tensor(option1_mask,dtype=torch.long))
                # all_option2_mask.append(torch.tensor(option2_mask,dtype=torch.long))
                # all_option3_mask.append(torch.tensor(option3_mask,dtype=torch.long))
                # all_option4_mask.append(torch.tensor(option4_mask,dtype=torch.long))
                # label )
                all_label.append(instance['label'])
        if('train' in each_file):
            train_dataset = TensorDataset(all_qa0_ids,all_qa1_ids,all_qa2_ids,all_qa3_ids,all_qa4_ids,all_len,all_label)
        else:
            dev_dataset = TensorDataset(all_qa0_ids,all_qa1_ids,all_qa2_ids,all_qa3_ids,all_qa4_ids,all_len,all_label)
            test_dataset = TensorDataset(all_qa0_ids,all_qa1_ids,all_qa2_ids,all_qa3_ids,all_qa4_ids,all_len)


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

