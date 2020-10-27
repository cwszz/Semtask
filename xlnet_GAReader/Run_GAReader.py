# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torchtext import data, datasets, vocab
from tqdm import tqdm, trange
from transformers import XLNetTokenizer, XLNetForMultipleChoice

from xlnet_GAReader.Utils.bert_embedding_utils import load_data
from xlnet_GAReader.Utils.utils import (classifiction_metric, epoch_time,
                                       get_device, word_tokenize)

logger = logging.getLogger(__name__)
def train(epoch_num, model, train_dataloader, dev_dataloader, optimizer, criterion, label_list, out_model_file, log_dir,
          print_step, clip,device):
    model.train()
    logging_dir = log_dir + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    outputting_dir = output_dir + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    # writer = SummaryWriter(log_dir=logging_dir)
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)
    
    global_step = 0
    best_dev_loss = float('inf')
    best_acc = 0.0

    for epoch in range(int(epoch_num)):
        print(f'---------------- Epoch: {epoch + 1:02} ----------')

        epoch_loss = 0
        train_steps = 0

        all_preds = np.array([], dtype=int)
        all_labels = np.array([], dtype=int)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            optimizer.zero_grad()
            batch = tuple(torch.tensor(t).to(device) for t in batch)
            max_len = torch.max(batch[5])
            # max_len = 300
            # if(max_len>1252):
            #     max_len = 1252
            input_ids= torch.cat([batch[0][:,0:max_len].unsqueeze(0),batch[1][:,0:max_len].unsqueeze(0),batch[2][:,0:max_len].unsqueeze(0)
            ,batch[3][:,0:max_len].unsqueeze(0),batch[4][:,0:max_len].unsqueeze(0)],dim=0).transpose(1,0)
            loss,logits = model(input_ids=input_ids,labels= batch[6])[:2]
            # loss = criterion(logits.view(-1, len(label_list)), batch[6])

            labels = batch[6].detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            global_step += 1

            epoch_loss += loss.item()
            train_steps += 1

            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)

            if global_step % print_step == 0:

                train_loss = epoch_loss / train_steps
                train_acc, train_report = classifiction_metric(
                    all_preds, all_labels, label_list)

                dev_loss, dev_acc, dev_report = evaluate(
                    model, dev_dataloader, criterion, label_list,device)
                c = global_step // print_step
                with open(logging_dir+'/log.txt','a+') as writer:
                    writer.write("loss/train   " + str(train_loss)[0:6] + "   " + str(c) )
                    writer.write("  loss/dev    " + str(dev_loss)[0:6] + "   " + str(c) )
                    writer.write("  acc/train   " + str(train_acc)[0:6] + "   " + str(c) )
                    writer.write("  acc/dev     " + str(dev_acc)[0:6] + "   " + str(c) +  '\n')
                # writer.add_scalar("loss/train", train_loss, c)
                # writer.add_scalar("loss/dev", dev_loss, c)

                # writer.add_scalar("acc/train", train_acc, c)
                # writer.add_scalar("acc/dev", dev_acc, c)

                for label in label_list:
                    with open(logging_dir+'/log.txt','a+') as writer:
                        writer.write("  " + str(label) + " : f1/train   " + str(train_report[label]['f1-score'])[0:6] + "   " + str(c) )
                        writer.write("  " + str(label) + " : f1/dev   " + str(dev_report[label]['f1-score'])[0:6] + "   " + str(c))
                    # writer.add_scalar(label + ":" + "f1/train",
                    #                   train_report[label]['f1-score'], c)
                    # writer.add_scalar(label + ":" + "f1/dev",
                    #                   dev_report[label]['f1-score'], c)
                        writer.write('\n')
                # print_list = ['macro avg', 'weighted avg']
                # for label in print_list:
                #     writer.write(str(label) + " : f1/train   " + str(train_report[label]['f1-score']) + "   " + str(c)+ '\n' )
                #     writer.write(str(label) + " : f1/dev   " + str(dev_report[label]['f1-score']) + "   " + str(c) + '\n')
                    # writer.add_scalar(label + ":" + "f1/train",
                    #                   train_report[label]['f1-score'], c)
                    # writer.add_scalar(label + ":" + "f1/dev",
                    #                   dev_report[label]['f1-score'], c)

                # if dev_loss < best_dev_loss:
                #     best_dev_loss = dev_loss

                if dev_acc > best_acc:
                    best_acc = dev_acc
                    torch.save(model.state_dict(),  outputting_dir)

                model.train()

    # writer.close()


def evaluate(model, iterator, criterion, label_list,device):
    model.eval()

    epoch_loss = 0

    all_preds = np.array([], dtype=int)
    all_labels = np.array([], dtype=int)

    with torch.no_grad():
        for batch in tqdm(iterator):
            with torch.no_grad():
                batch = tuple(torch.tensor(t).to(device) for t in batch)
                max_len = torch.max(batch[5])
                # if(max_len>1252):
                #     max_len = 1252
                input_ids= torch.cat([batch[0][:,0:max_len].unsqueeze(0),batch[1][:,0:max_len].unsqueeze(0),batch[2][:,0:max_len].unsqueeze(0)
                ,batch[3][:,0:max_len].unsqueeze(0),batch[4][:,0:max_len].unsqueeze(0)],dim=0).transpose(1,0)
                loss,logits = model(input_ids=input_ids,labels= batch[6])

            # loss = criterion(logits.view(-1, len(label_list)), batch[5])

            labels = batch[6].detach().cpu().numpy()
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

            all_preds = np.append(all_preds, preds)
            all_labels = np.append(all_labels, labels)
            epoch_loss += loss.item()

    acc, report = classifiction_metric(
        all_preds, all_labels, label_list)

    return epoch_loss / len(iterator), acc, report


def main(config, model_filename):
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    if not os.path.exists(config.cache_dir):
        os.makedirs(config.cache_dir)

    model_file = os.path.join(
        config.output_dir, model_filename)

    # Prepare the device
    # gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
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
    model = XLNetForMultipleChoice.from_pretrained('./xlnet_model') # ./xlnet_model

    cache_train_dataset = "cached_dataset_train"
    cache_dev_dataset = "cached_dataset_dev"
    if os.path.exists(config.cache_dir + '/' + cache_train_dataset):
        logger.info("Loading features from cached file %s", config.cache_dir + '/' + cache_train_dataset)
        train_dataset = torch.load(config.cache_dir + '/' + cache_train_dataset)
        dev_dataset = torch.load(config.cache_dir + '/' + cache_dev_dataset)
    else:
        train_dataset, dev_dataset, test_dataset = load_data(config.data_path,  device, tokenizer, config.cache_dir,128,1024)
        logger.info("save cached file in  %s", config.cache_dir)
        torch.save(train_dataset,config.cache_dir + '/' + cache_train_dataset)
        torch.save(dev_dataset,config.cache_dir + '/' + cache_dev_dataset)
    train_sampler = RandomSampler(train_dataset)
    dev_sampler =RandomSampler(dev_dataset)
    train_dataloader  = DataLoader(train_dataset,shuffle=True,sampler= train_sampler,batch_size= config.train_batch_size,num_workers=8,pin_memory=False)
    dev_dataloader  = DataLoader(dev_dataset,shuffle=True,sampler= dev_sampler,batch_size= config.dev_batch_size,num_workers=8,pin_memory=False)
    # train_iterator = trange(int(config.epoch_num))
    # if config.model_name == "GAReader":
    #     from Bert_GAReader.GAReader.GAReader import GAReader
    #     model = GAReader(
    #         config.bert_word_dim, config.output_dim, config.hidden_size,
    #         config.rnn_num_layers, config.ga_layers, config.bidirectional,
    #         config.dropout, bert_config)
    #     print(model)
    # no_decay = ['bias', 'LayerNorm.weight']

    # optimizer = optim.Adam(model.parameters(), lr=config.lr)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    # optimizer = optim.AdamW(optimizer_grouped_parameter,lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    if config.do_train:
        train(config.epoch_num, model, train_dataloader, dev_dataloader, optimizer, criterion, ['0', '1', '2', '3', '4'],
              model_file, config.log_dir, config.print_step, config.clip,device)

    model.load_state_dict(torch.load(model_file))

    test_loss, test_acc, test_report = evaluate(
        model, dev_dataloader, criterion, ['0', '1', '2', '3'],device)
    print("-------------- Test -------------")
    print("\t Loss: {} | Acc: {} | Macro avg F1: {} | Weighted avg F1: {}".format(
        test_loss, test_acc, test_report['macro avg']['f1-score'], test_report['weighted avg']['f1-score']))


if __name__ == "__main__":

    model_name = "GAReader"
    data_dir = "./data/1task_train"
    embedding_folder = "./ga/cache/vocabulary/"

    output_dir = "./ga/xlnet_output/"
    cache_dir = "./ga/xlnet_cache/"
    log_dir = "./ga/xlnet_log/"
    bert_config_path = "./xlnet_model/"
    
    model_filename = "model_adam1.pt"
    

    if model_name == "GAReader":
        from xlnet_GAReader.GAReader import GAReader, args


        main(args.get_args(data_dir, cache_dir, embedding_folder, output_dir, log_dir,bert_config_path), model_filename)
