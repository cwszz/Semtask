# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import time
# from apex import amp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torchtext import data, datasets, vocab
from tqdm import tqdm, trange
from pytorch_pretrained_bert.optimization import BertAdam
from transformers import BertConfig, BertTokenizer,AdamW,get_linear_schedule_with_warmup

from Bert_sentences_class.Utils.bert_embedding_utils import load_data,collate_fn
from Bert_sentences_class.Utils.utils import (classifiction_metric, epoch_time,
                                       get_device, word_tokenize)

logger = logging.getLogger(__name__)
def train(epoch_num, model, train_dataloader, dev_dataloader, optimizer, criterion, label_list, out_model_file, log_dir,
          print_step, clip,device,experiment_detail,scheduler):
    model.train()
    logging_dir = log_dir + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    outputting_dir = output_dir + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    # writer = SummaryWriter(log_dir=logging_dir)
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)
    with open(logging_dir + '/experment_setting.txt','a+') as f:
        f.write(experiment_detail)
    global_step = 0
    best_dev_loss = float('inf')
    best_acc = 0.0

    for epoch in range(int(epoch_num)):
        print(f'---------------- Epoch: {epoch + 1:02} ----------')

        epoch_loss = 0
        train_steps = 0
        step_loss = 0
        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            optimizer.zero_grad()
            batch_part = [torch.tensor(t).to(device) for t in batch if not isinstance(t,list)]
            batch_part.append([t.to(device) for t in batch[6]])
            logits = model(batch_part)

            loss = criterion(logits.view(-1, len(label_list)), batch_part[6])

            labels = batch_part[6].detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            step_loss += loss.item()
            if(step % 10 == 0):
                with open(logging_dir+'/loss_step.txt','a+') as f:
                    f.write("--------------------------step"+ str(step)+ "--------------------------\n")
                    f.write(str(step_loss/10)+'\n')
                    step_loss =0
                with open(logging_dir + '/train_logit.txt','a+') as f:
                    f.write("--------------------------step"+ str(step)+ "--------------------------\n")
                    f.write(str(logits)+'\n')
                with open(logging_dir + '/weight.txt','a+') as f:
                    f.write("--------------------------step"+ str(step)+ "--------------------------\n")
                    para = [{'name':n,'para':p}  for n,p in model.named_parameters() if 'weight' in n and 'bert' not in n]
                    f.write(str(para) + '\n')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()
            train_steps += 1

            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)

            if global_step % print_step == 0:

                train_loss = epoch_loss / train_steps
                train_acc, train_report = classifiction_metric(
                    all_preds, all_labels, label_list)

                # with torch.no_grad():
                dev_loss, dev_acc, dev_report = evaluate(
                    model, dev_dataloader, criterion, label_list,device,logging_dir)
                c = global_step // print_step
                with open(logging_dir+'/log.txt','a+') as writer:
                    writer.write("loss/train   " + str(train_loss)[0:6] + "   " + str(c) )
                    writer.write("  loss/dev    " + str(dev_loss)[0:6] + "   " + str(c) +'\n')
                    writer.write("  acc/train   " + str(train_acc)[0:6] + "   " + str(c) )
                    writer.write("  acc/dev     " + str(dev_acc)[0:6] + "   " + str(c) +'\n' )
               

                for label in label_list:
                    with open(logging_dir+'/log.txt','a+') as writer:
                        writer.write("  " + str(label) + " : f1/train   " + str(train_report[label]['f1-score'])[0:6] + "   " + str(c) )
                        writer.write("  " + str(label) + " : f1/dev   " + str(dev_report[label]['f1-score'])[0:6] + "   " + str(c))
                        writer.write('\n')
              

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    torch.save(model.state_dict(),  outputting_dir+'best_model')

                model.train()

    # writer.close()


def evaluate(model, iterator, criterion, label_list,device,log_dir):
    model.eval()

    epoch_loss = 0

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(iterator):
            with torch.no_grad():
                batch_part = [torch.tensor(t).to(device) for t in batch if not isinstance(t,list)]
                batch_part.append([t.to(device) for t in batch[6]])
                logits = model(batch_part)
            cnt += 1
            loss = criterion(logits.view(-1, len(label_list)), batch_part[6])
            if cnt % 10 == 1:
                with open(log_dir + '/test_logits.txt','a+') as f:
                    f.write(str(logits) + '\n')
            labels = batch_part[6].detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)
            epoch_loss += loss.item()

    acc, report = classifiction_metric(
        all_preds, all_labels, label_list)
    with open(log_dir+'/prediction.txt','a+') as f:
        f.write(str(all_preds) + '\n')
        f.write("_______________\n")
        f.write(str(all_labels) +'\n')
    return epoch_loss / len(iterator), acc, report


def main(config, model_filename):
    # device_id = 0
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if not os.path.exists(config.cache_dir):
        os.makedirs(config.cache_dir)

    model_file = os.path.join(
        config.output_dir, model_filename)

    # Prepare the device
    gpu_ids = [2,3,4]
    device, n_gpu = get_device(gpu_ids[0])
    if n_gpu > 1:
        n_gpu = len(gpu_ids)

    # Set Random Seeds
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True

    tokenizer = BertTokenizer.from_pretrained(config.bert_config_path,do_lower_case=True)
    bert_config = BertConfig.from_pretrained(config.bert_config_path)

    cache_train_dataset = "cached_dataset_train_DCMN"
    cache_dev_dataset = "cached_dataset_dev_DCMN"
    if os.path.exists(config.cache_dir + '/' + cache_train_dataset):
        logger.info("Loading features from cached file %s", config.cache_dir + '/' + cache_train_dataset)
        train_dataset = torch.load(config.cache_dir + '/' + cache_train_dataset)
        dev_dataset = torch.load(config.cache_dir + '/' + cache_dev_dataset)
    else:
        train_dataset, dev_dataset, test_dataset = load_data(config.data_path,  device, tokenizer, config.cache_dir,96,128,5)
        logger.info("save cached file in  %s", config.cache_dir)
        torch.save(train_dataset,config.cache_dir + '/' + cache_train_dataset)
        torch.save(dev_dataset,config.cache_dir + '/' + cache_dev_dataset)
    print(len(train_dataset))
    train_sampler = RandomSampler(train_dataset)
    dev_sampler =RandomSampler(dev_dataset)
    train_dataloader  = DataLoader(train_dataset,sampler= train_sampler,batch_size= config.train_batch_size,num_workers=4,pin_memory=False,collate_fn=collate_fn)
    dev_dataloader  = DataLoader(dev_dataset,sampler= dev_sampler,batch_size= config.dev_batch_size,num_workers=4,pin_memory=False,collate_fn=collate_fn)
    # train_iterator = trange(int(config.epoch_num))
    if config.model_name == "GAReader":
        from Bert_sentences_class.GAReader.GAReader import GAReader
        model = GAReader(
            config.bert_word_dim, config.output_dim, config.hidden_size,
            config.rnn_num_layers, config.ga_layers, config.bidirectional,
            config.dropout, bert_config)
        # trained_file = './ga/output/2020-10-13-23_07_26'
        # tt = torch.load(trained_file)
        # model.load_state_dict(torch.load(trained_file,map_location={'cuda:2':'cuda:1'}))
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    part_1_1_lr,part_1_2_lr,part_2_1_lr,part_2_2_lr = 1e-4, 1e-4,1e-4,1e-4
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay) and 'bert' not in n ] , 'weight_decay': 0.01,'lr':part_1_1_lr,
        'name':[n for n, p in param_optimizer if not any(
            nd in n for nd in no_decay) and 'bert' not in n ] },
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay) and 'bert' not in n ], 'weight_decay': 0.0,'lr':part_1_2_lr,
        'name':[n for n, p in param_optimizer if  any(
            nd in n for nd in no_decay) and 'bert' not in n ] },
        # {'params': [p for n, p in param_optimizer if not any(
        #     nd in n for nd in no_decay) and 'bert' not in n  , 'weight_decay': 0.01,'lr':part_2_1_lr,
        # 'name':[n for n, p in param_optimizer if not any(
        #     nd in n for nd in no_decay) and 'bert' not in n ]},
        # {'params': [p for n, p in param_optimizer if any(
        #     nd in n for nd in no_decay) and 'bert' not in n , 'weight_decay': 0.0,'lr':part_2_2_lr,
        # 'name':[n for n, p in param_optimizer if  any(
        #     nd in n for nd in no_decay) and 'bert' not in n ]},
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay) and 'bert'  in n], 'weight_decay': 0.01,
        'name':[n for n, p in param_optimizer if not any(
            nd in n for nd in no_decay) and 'bert'  in n ]},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay) and 'bert'  in n], 'weight_decay': 0.0,
        'name':[n for n, p in param_optimizer if  any(
            nd in n for nd in no_decay) and 'bert'  in n ]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=config.lr)
    # optimizer = BertAdam(optimizer_grouped_parameters,lr=config.lr,warmup=1e-3,t_total=1e5)
    # model,optimizer = amp.initialize(model,optimizer,opt_level="01")
    scheduler = get_linear_schedule_with_warmup(optimizer,16000,200000)
    
    criterion = nn.CrossEntropyLoss()
    # 并行GPU
    model = torch.nn.DataParallel(model,gpu_ids)
    model = model.to(device)
    criterion = criterion.to(device)
    experiment_detail = str(config.dropout) + '\n' + str(config.lr)+','+ str(part_1_1_lr)+','+str(part_1_2_lr)+','+str(part_2_1_lr)+','+str(part_2_2_lr) + '\n' + str(config.train_batch_size) + "\n" + "layers:7"
    if config.do_train:
        train(config.epoch_num, model, train_dataloader, dev_dataloader, optimizer, criterion, ['0', '1', '2', '3', '4'],
              model_file, config.log_dir, config.print_step, config.clip,device,experiment_detail,scheduler)

    model.load_state_dict(torch.load(model_file))

    test_loss, test_acc, test_report = evaluate(
        model, dev_dataloader, criterion, ['0', '1', '2', '3','4'],device,log_dir)
    print("-------------- Test -------------")
    print("\t Loss: {} | Acc: {} | Macro avg F1: {} | Weighted avg F1: {}".format(
        test_loss, test_acc, test_report['macro avg']['f1-score'], test_report['weighted avg']['f1-score']))


if __name__ == "__main__":

    model_name = "GAReader"
    data_dir = "./data/1task_train"
    embedding_folder = "./ga/cache/vocabulary/"

    output_dir = "./ga/class_output/"
    cache_dir = "./ga/class_cache/"
    log_dir = "./ga/class_log/"
    bert_config_path = "./new_bert"
    
    model_filename = "model_adam1.pt"
    

    if model_name == "GAReader":
        from Bert_sentences_class.GAReader import GAReader, args


        main(args.get_args(data_dir, cache_dir, embedding_folder, output_dir, log_dir,bert_config_path), model_filename)
