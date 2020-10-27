# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torchtext import data, datasets, vocab
from tqdm import tqdm, trange
# from transformers import BertConfig, BertTokenizer,AdamW,get_linear_schedule_with_warmup
from transformers import (AdamW, XLNetConfig, XLNetTokenizer,
                          get_linear_schedule_with_warmup)

from XLNet_Linear.Utils.bert_embedding_utils import load_data
from XLNet_Linear.Utils.utils import (classifiction_metric, epoch_time,
                                      get_device, word_tokenize)

logger = logging.getLogger(__name__)
def train(epoch_num, model, train_dataloader, dev_dataloader, optimizer, criterion, label_list, out_model_file, log_dir,
          print_step, clip,device,scheduler):
    model.train()
    logging_dir = log_dir + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    outputting_dir = output_dir + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    # writer = SummaryWriter(log_dir=logging_dir)
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)
    if not os.path.exists(outputting_dir):
        os.mkdir(outputting_dir)
    
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
            batch = tuple(torch.tensor(t).to(device) for t in batch)
            logits = model(batch)
            if(batch[6].size(0)==1):
                logits = logits[0].unsqueeze(0)
            loss = criterion(logits.view(-1, len(label_list)), batch[6])

            labels = batch[6].detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            step_loss += loss.item()
            if(step % 10 == 0):
                with open(logging_dir+'/loss_step.txt','a+') as f:
                    f.write("--------------------------step"+ str(step)+ "--------------------------\n")
                    f.write(str(step_loss/10)+'\n')
                    step_loss =0
                with open(logging_dir+'/para.txt','a+') as f:
                    f.write("--------------------------step"+ str(step)+ "--------------------------\n")
                    f.write(str(model.final_liear.weight)+'\n')
            # if(step % print_step == 0):
            #     with open(logging_dir+'/para.txt','a+') as f:
            #         f.write("--------------------------step"+ str(step)+ "--------------------------\n")
            #         for name,para in model.named_parameters():
            #             if('rnn' in name and 'weight' in name):
            #                     f.write(str(name) + ' : '+str(para)+'\n')
            loss.backward()
            # if(step % print_step == 0):
            #     with open(logging_dir+'/para.txt','a+') as f:
            #         for name,para in model.named_parameters():
            #             if('rnn' in name and 'weight' in name):
            #                 f.write(str(name) + ' : '+str(para)+'\n')
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
                    writer.write("  loss/dev    " + str(dev_loss)[0:6] + "   " + str(c) )
                    writer.write("  acc/train   " + str(train_acc)[0:6] + "   " + str(c) )
                    writer.write("  acc/dev     " + str(dev_acc)[0:6] + "   " + str(c) +'\n' )
                # writer.add_scalar("loss/train", train_loss, c)
                # writer.add_scalar("loss/dev", dev_loss, c)

                # writer.add_scalar("acc/train", train_acc, c)
                # writer.add_scalar("acc/dev", dev_acc, c)

                for label in label_list:
                    with open(logging_dir+'/log.txt','a+') as writer:
                        writer.write("  " + str(label) + " : f1/train   " + str(train_report[label]['f1-score'])[0:6] + "   " + str(c) )
                        writer.write("  " + str(label) + " : f1/dev   " + str(dev_report[label]['f1-score'])[0:6] + "   " + str(c))
                        writer.write('\n')
              

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    torch.save(model.state_dict(),  outputting_dir+'/best_model_linear')
                torch.save(model.state_dict(),  outputting_dir+'/last_model_linear')
                model.train()

    # writer.close()


def evaluate(model, iterator, criterion, label_list,device,log_dir):
    model.eval()

    epoch_loss = 0

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)

    with torch.no_grad():
        for batch in tqdm(iterator):
            with torch.no_grad():
                batch = tuple(torch.tensor(t).to(device) for t in batch)
                logits = model(batch)
            if(batch[6].size(0)==1):
                logits = logits[0].unsqueeze(0)
            loss = criterion(logits.view(-1, len(label_list)), batch[6])

            labels = batch[6].detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)
            epoch_loss += loss.item()

    acc, report = classifiction_metric(
        all_preds, all_labels, label_list)
    with open(log_dir+'/prediction.txt','a+') as f:
        result = Counter(all_preds)
        golden_result = Counter(all_labels)
        f.write(str(result) + '\n')
        f.write(str(golden_result) + '\n')
    return epoch_loss / len(iterator), acc, report


def main(config, model_filename):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if not os.path.exists(config.cache_dir):
        os.makedirs(config.cache_dir)
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    model_file = os.path.join(
        config.output_dir, model_filename)

    # Prepare the device
    gpu_ids = [2]
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

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    xlnet_config = XLNetConfig.from_pretrained(config.bert_config_path)

    cache_train_dataset = "cached_dataset_train_linear_512"
    cache_dev_dataset = "cached_dataset_dev_linear_512"
    if os.path.exists(config.cache_dir + '/' + cache_train_dataset):
        logger.info("Loading features from cached file %s", config.cache_dir + '/' + cache_train_dataset)
        train_dataset = torch.load(config.cache_dir + '/' + cache_train_dataset)
        dev_dataset = torch.load(config.cache_dir + '/' + cache_dev_dataset)
    else:
        train_dataset, dev_dataset, test_dataset = load_data(config.data_path,  device, tokenizer, config.cache_dir,64,960)
        logger.info("save cached file in  %s", config.cache_dir)
        torch.save(train_dataset,config.cache_dir + '/' + cache_train_dataset)
        torch.save(dev_dataset,config.cache_dir + '/' + cache_dev_dataset)
    # train_sampler = RandomSampler(train_dataset)
    # dev_sampler =RandomSampler(dev_dataset)
    train_dataloader  = DataLoader(train_dataset,shuffle=True,batch_size= config.train_batch_size,num_workers=8,pin_memory=False,drop_last=False )
    dev_dataloader  = DataLoader(dev_dataset,shuffle=True, batch_size= config.dev_batch_size,num_workers=8,pin_memory=False,drop_last=False )
    # train_iterator = trange(int(config.epoch_num))
    if config.model_name == "GAReader":
        from XLNet_Linear.GAReader.GAReader import GAReader
        model = GAReader(
            config.bert_word_dim, config.output_dim, config.hidden_size,
            config.rnn_num_layers, config.ga_layers, config.bidirectional,
            config.dropout, xlnet_config)
        
    # optimizer_grouped_parameter = [
    #     {'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'embedding' not in n and 'bert' not in n]},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'embedding' not in n and 'bert' not in n]}
    # ]
    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    # optimizer_parameter =[
    #     {'params':model.word_embedding.bert.parameters()},
    #     {'params':model.word_embedding.aggregation.parameters(),'lr':1e-4},
    #     # {'params':model.rnn.parameters(),'lr':1e-3},
    #     # {'params':model.ga_rnn.parameters(),'lr':1e-3},
    #     # {'params':model.mlp_att.parameters(),'lr':1e-2},
    #     # {'params':model.dot_layer.parameters(),'lr':1e-2},
    #     {'params':model.final_liear.parameters(),'lr':1e-4},
    # ]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay) and 'xlnet' not in n],
        'name':[n for n, p in param_optimizer if not any(
            nd in n for nd in no_decay) and 'xlnet' not in n], 
        'weight_decay': 0.01,'lr':3e-4},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay) and 'xlnet' not in n],
        'name':[n for n, p in param_optimizer if not any(
            nd in n for nd in no_decay) and 'xlnet' not in n],  
        'weight_decay': 0.0,'lr':3e-4},
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay) and 'xlnet' in n],
        'name':[n for n, p in param_optimizer if not any(
            nd in n for nd in no_decay) and 'xlnet' not in n], 
        'weight_decay': 0.01,'lr':config.lr},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay) and 'xlnet'  in n],
        'name':[n for n, p in param_optimizer if not any(
            nd in n for nd in no_decay) and 'xlnet' not in n],  
        'weight_decay': 0.0,'lr':config.lr}
    ]
    # print(optimizer_grouped_parameter)
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.lr,eps=1e-6)
    # optimizer = optim.SGD(model.parameters(), lr=config.lr)
    # print(optimizer_grouped_parameter)
    # optimizer = optim.SGD(optimizer_parameter,lr=config.lr)
    # model,optimizer = amp.initialize(model,optimizer,opt_level="01")
    scheduler = get_linear_schedule_with_warmup(optimizer,16000,200000)
    
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    if config.do_train:
        train(config.epoch_num, model, train_dataloader, dev_dataloader, optimizer, criterion, ['0', '1', '2', '3', '4'],
              model_file, config.log_dir, config.print_step, config.clip,device,scheduler)
    # trained_file = './ga/output/2020-10-20-22_41_37best_model_linear'
    # tt = torch.load(trained_file)
    # model.load_state_dict(torch.load(trained_file,map_location={'cuda:2':'cuda:1'}))
    model.load_state_dict(torch.load(model_file))

    test_loss, test_acc, test_report = evaluate(
        model, train_dataloader, criterion, ['0', '1', '2', '3','4'],device,log_dir)
    print("-------------- Test -------------")
    print("\t Loss: {} | Acc: {} | Macro avg F1: {} | Weighted avg F1: {}".format(
        test_loss, test_acc, test_report['macro avg']['f1-score'], test_report['weighted avg']['f1-score']))


if __name__ == "__main__":

    model_name = "GAReader"
    data_dir = "./data/1task_train"
    embedding_folder = "./ga/cache/vocabulary/"

    output_dir = "./ga/xl-linear_output/"
    cache_dir = "./ga/xl-linear_cache/"
    log_dir = "./ga/xl-linear_log/"
    bert_config_path = "./xlnet_model"
    
    model_filename = "model_adam1.pt"
    

    if model_name == "GAReader":
        from XLNet_Linear.GAReader import GAReader, args


        main(args.get_args(data_dir, cache_dir, embedding_folder, output_dir, log_dir,bert_config_path), model_filename)
