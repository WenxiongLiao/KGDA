import imp
import json 
import re
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd
from datasets import load_dataset
import torch.nn as nn
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import random
import itertools
from transformers import AutoTokenizer,BertTokenizer, BertModel,AutoModel,BertForMaskedLM,DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from transformers import pipeline
from datasets import load_metric
from sklearn.metrics import classification_report,accuracy_score

from data_processing.data_utils import read_common_words
from  config import config_param


def global_init():
    global device,tokenizer,metric,data_collator,common_words
    device = torch.device(config_param.device) if torch.cuda.is_available() else torch.device("cpu")
    if Path(config_param.relation_model_save_dir).exists():
        tokenizer = AutoTokenizer.from_pretrained(config_param.relation_model_save_dir)#,add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_param.model_checkpoint)#,add_prefix_space=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    common_words = read_common_words(config_param.common_words_path)


def sample_to_relations(examples):
    '''
    Convert a row sample of csv to relations sentence
    '''
    entity_label_words = list(config_param.entity_label_word_dict.keys())
    all_tokens = examples["tokens"]
    all_relations = examples["relations"]
    new_sample_tokens = []
    new_labels = []

    mention_entity_label_words_dict = {}
    for relations in all_relations:
        relations = eval(relations)
        if len(relations) > 0:
            for relation in relations:
                mention_entity_label_words_dict[relation['head']] = relation['head_type']
                mention_entity_label_words_dict[relation['tail']] = relation['tail_type']

    assert len(all_tokens) == len(all_relations), print('dimension error!')
    
    for i in range(len(all_relations)):
        
        tokens = eval(all_tokens[i])
        relations = eval(all_relations[i])
        
        #build no_relation_entity_pair for Construct negative samples
        relation_entity_pairs = []
        all_entity = set()
        for relation_triplet in relations:
            relation_entity_pairs.append(relation_triplet['head'].strip() + '_' + relation_triplet['tail'].strip())
            all_entity.add(relation_triplet['head'].strip())
            all_entity.add(relation_triplet['tail'].strip())
        combine_entity = list(itertools.permutations(all_entity, 2))
        all_entity_pairs = [head + '_' + tail for (head,tail) in combine_entity]
        no_relation_entity_pairs = list(set(all_entity_pairs) - set(relation_entity_pairs))
        
        for relation_triplet in relations:

            head_entity = relation_triplet['head'].strip().split()
            tail_entity = relation_triplet['tail'].strip().split()
            head_type = relation_triplet['head_type'].strip().split()
            tail_type = relation_triplet['tail_type'].strip().split()
            label = relation_triplet['relation']
            if config_param.with_entity_type == False:
                sample = 'head : '.split() + head_entity + ' [SEP] tail : '.split() + tail_entity + [' [SEP] '] + tokens
            else:
                sample = 'head : '.split() + head_entity + ' ( '.split() + head_type +  ' ) '.split()  \
                        + ' [SEP] tail : '.split() + tail_entity + ' ( '.split() + tail_type +  ' ) '.split()  \
                        + [' [SEP] '] + tokens
            sample = ' '.join(sample)
            new_sample_tokens.append(sample)
            new_labels.append(label)
            
            #Construct negative samples
            if random.random() < config_param.relation_negative_ratio:
                #Construct negative samples
                if random.random() < config_param.relation_common_negative_ratio:
                    #Construct negative samples with common words
                    sentence_common_words = list(set(tokens) & set(common_words))
                    if len(sentence_common_words) == 0:
                        continue
                    entity = random.choice(sentence_common_words)
                    if random.random() < 0.5:
                        head_entity = entity.strip()
                        tail_entity = relation_triplet['tail'].strip()
                        head_type = random.choice(entity_label_words)
                        tail_type = relation_triplet['tail_type'].strip()
                    else:
                        head_entity = relation_triplet['head'].strip()
                        tail_entity = entity.strip()
                        head_type = relation_triplet['head_type'].strip()
                        tail_type = random.choice(entity_label_words)
                    
                else:
                    #Construct negative samples with no relation enetities
                    if len(no_relation_entity_pairs) == 0:
                        continue
                    no_relation_entity_pair = random.choice(no_relation_entity_pairs)
                    [head_entity,tail_entity] = no_relation_entity_pair.split('_')[0:2]
                    head_type = mention_entity_label_words_dict[head_entity].strip()
                    tail_type = mention_entity_label_words_dict[tail_entity].strip()
                    
                head_entity = head_entity.split()
                tail_entity = tail_entity.split()
                label = config_param.relation_label2id['NULL']
                if config_param.with_entity_type == False:
                    sample = 'head : '.split() + head_entity + ' [SEP] tail : '.split() + tail_entity + [' [SEP] '] + tokens
                else:
                    sample = 'head : '.split() + head_entity + ' ( '.split() + head_type.strip().split() +  ' ) '.split()  \
                            + ' [SEP] tail : '.split() + tail_entity + ' ( '.split() + tail_type.strip().split() +  ' ) '.split()  \
                            + [' [SEP] '] + tokens
                sample = ' '.join(sample)
                new_sample_tokens.append(sample)
                new_labels.append(label)

                            
    

    new_examples = {}
    new_examples["tokens"] = new_sample_tokens
    new_examples["labels"] = new_labels

    return new_examples


def tokenize_function(example):
    return tokenizer(example["tokens"], truncation=True,max_length = 510)

class Relation():
    def __init__(self) -> None:
        global_init()
        self.softmax = nn.Softmax(dim = 1)
        self.model = None
        # self.relation_classifier = None

    def relation_model_init(self):
        if Path(config_param.relation_model_save_dir).exists():
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config_param.relation_model_save_dir, num_labels=config_param.relation_num_labels)

        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config_param.model_checkpoint, num_labels=config_param.relation_num_labels)
        #self.model = nn.DataParallel(self.model,device_ids=[0,1,2])
        self.model.to(device)
    
    def get_train_val_dataLoader(self,data_path):
        '''
        load single csv file and split as train_dataloader , eval_dataloader
        '''
        raw_datasets = load_dataset("csv", data_files=data_path)

        raw_datasets = raw_datasets['train'].train_test_split(test_size=config_param.relation_test_ratio, shuffle=True, seed=2022)

        raw_datasets = raw_datasets.map(
            sample_to_relations,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['tokens'])

        train_dataloader = DataLoader(
            tokenized_datasets["train"], shuffle=True, batch_size=config_param.relation_batch_size, collate_fn = data_collator
        )
        eval_dataloader = DataLoader(
            tokenized_datasets["test"], batch_size=config_param.relation_batch_size, collate_fn = data_collator
        )

        return train_dataloader,eval_dataloader
    
    def get_test_dataLoader(self,data_path):
        '''
        load single csv file and split as test_dataLoader
        '''
        raw_datasets = load_dataset("csv", data_files=data_path)
        old_relation_negative_ratio =  config_param.relation_negative_ratio 
        config_param.relation_negative_ratio = -1

        raw_datasets = raw_datasets.map(
            sample_to_relations,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(['tokens'])

        test_dataLoader = DataLoader(
            tokenized_datasets["train"], shuffle=True, batch_size=config_param.relation_batch_size, collate_fn = data_collator
        )

        config_param.relation_negative_ratio = old_relation_negative_ratio

        return test_dataLoader


    def train_relation_model(self,data_path):
        # init relation_model
        print('model init...')
        self.relation_model_init()

        #get dataLoader
        print('get dataLoader...')
        train_dataloader,eval_dataloader = self.get_train_val_dataLoader(data_path)

        optimizer = AdamW(self.model.parameters(), lr=config_param.relation_lr)

        # init lr
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = config_param.relation_num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        progress_bar = tqdm(range(num_training_steps))

        max_acc = 0
        print('model training...')
        for epoch in range(config_param.relation_num_train_epochs):
            self.model.train()
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                #loss.sum().backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            
            self.model.eval()
            y_true = []
            y_pred = []
            for batch in eval_dataloader:

                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                labels = batch["labels"].detach().cpu().numpy()

                if len(y_true) == 0:
                    y_true = labels
                    y_pred = predictions
                else:
                    y_true = np.concatenate([y_true,labels])
                    y_pred = np.concatenate([y_pred,predictions])

            if len(y_true) >0:
                print(classification_report(y_true, y_pred,digits= 4))
                acc = accuracy_score(y_true, y_pred)
                if  acc > max_acc:
                    max_acc = acc
                    print('save model')
                    self.model.save_pretrained(config_param.relation_model_save_dir)
                    #self.model.module.save_pretrained(config_param.relation_model_save_dir)
        tokenizer.save_pretrained(config_param.relation_model_save_dir)
        # self.relation_classifier = None
        
        torch.cuda.empty_cache()

    
    def test_relation_model(self,data_path):
        # init relation_model
        print('model init...')
        self.relation_model_init()

        #get dataLoader
        print('get dataLoader...')
        test_dataLoader = self.get_test_dataLoader(data_path)
        self.model.eval()
        y_true = []
        y_pred = []
        for batch in tqdm(test_dataLoader):

            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()

            if len(y_true) == 0:
                y_true = labels
                y_pred = predictions
            else:
                y_true = np.concatenate([y_true,labels])
                y_pred = np.concatenate([y_pred,predictions])
        print(classification_report(y_true, y_pred,digits= 4))

        torch.cuda.empty_cache()


    def relation_pred_batch(self,head_entity_list,tail_entity_list,sentences,head_entity_type_list =None,tail_entity_type_list =None):
        
        batch_size = config_param.relation_pred_batch_size
        assert len(head_entity_list) == len(tail_entity_list) and len(tail_entity_list) == len(sentences), print('dimension error!')
        if config_param.with_entity_type == True:
            assert head_entity_type_list!=None and tail_entity_type_list!=None, print('params error!')

        if self.model == None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config_param.relation_model_save_dir, num_labels=config_param.relation_num_labels)
            #self.model = nn.DataParallel(self.model,device_ids=[0,1,2])
            self.model.to(device)
        self.model.eval()

        sample_num = len(head_entity_list)
        batch_num = int(sample_num//batch_size) if sample_num%batch_size == 0 else int(sample_num/batch_size) + 1
        
        label_names = []
        label_ids = []
        probabilities = []
        for i in range(batch_num):
            head_entity_batch = head_entity_list[i * batch_size : (i +1 ) * batch_size]
            tail_entity_batch = tail_entity_list[i * batch_size : (i +1 ) * batch_size]
            if config_param.with_entity_type == True:
                head_entity_type_batch = head_entity_type_list[i * batch_size : (i +1 ) * batch_size]
                tail_entity_type_batch = tail_entity_type_list[i * batch_size : (i +1 ) * batch_size]

            sentence_batch = sentences[i * batch_size : (i +1 ) * batch_size]

            sample_batch = []
            for j in range(len(head_entity_batch)):
                if config_param.with_entity_type == False:
                    sample = 'head : '.split() + head_entity_batch[j].strip().split() \
                            + ' [SEP] tail : '.split() + tail_entity_batch[j].strip().split() \
                            + [' [SEP] '] + sentence_batch[j].strip().split()
                else:
                    sample = 'head : '.split() + head_entity_batch[j].strip().split() + ' ( '.split() + head_entity_type_batch[j].strip().split() +  ' ) '.split() \
                    + ' [SEP] tail : '.split() + tail_entity_batch[j].strip().split() + ' ( '.split() + tail_entity_type_batch[j].strip().split() +  ' ) '.split() \
                    + [' [SEP] '] + sentence_batch[j].strip().split()

                sample = ' '.join(sample)
                sample_batch.append(sample)
            

            inputs = tokenizer(sample_batch, return_tensors="pt", truncation=True, padding=True, max_length = 510)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                pred_logits = self.model(**inputs).logits
            pred_logits = self.softmax(pred_logits)

            label_id = torch.argmax(pred_logits,dim = 1).detach().cpu().numpy()
            probability = [pred_logits.detach().cpu().numpy()[i,label] for i, label in enumerate(label_id)]
            
            label_ids = np.concatenate([label_ids,label_id])
            probabilities = np.concatenate([probabilities,probability]) 
            label_names =  np.concatenate([label_names , [config_param.relation_id2label[label] for label in label_id]])

        torch.cuda.empty_cache()

        assert len(label_ids) == len(probabilities) and len(probabilities) == len(label_names), print('dimension error')

        result = {'label_name':list(label_names),'label_id':list(label_ids),'probability':list(probabilities)}

        return result


    def relation_pred(self,head_entity,tail_entity,sentence,with_entity_type ,head_entity_type=None,tail_entity_type=None):
        if with_entity_type == True:
            assert head_entity_type!=None and tail_entity_type!=None, print('params error!')

        if self.model == None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config_param.relation_model_save_dir, num_labels=config_param.relation_num_labels)
            #self.model = nn.DataParallel(self.model,device_ids=[0,1,2])
            self.model.to(device)
        self.model.eval()

        if with_entity_type == False:
            sample = 'head : '.split() + head_entity.strip().split() + ' [SEP] tail : '.split() + tail_entity.strip().split() + [' [SEP] '] + sentence.strip().split()
        else:
            sample = 'head : '.split() + head_entity.strip().split() + ' ( '.split() + head_entity_type.strip().split() +  ' ) '.split() \
                    + ' [SEP] tail : '.split() + tail_entity.strip().split() + ' ( '.split() + tail_entity_type.strip().split() +  ' ) '.split() \
                    + [' [SEP] '] + sentence.strip().split()

        sample = ' '.join(sample)
        print(sample)

        inputs = tokenizer(sample, return_tensors="pt", truncation=True, max_length = 510)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            pred_logits = self.model(**inputs).logits
        pred_logits = self.softmax(pred_logits)

        label_id = torch.argmax(pred_logits,dim = 1).detach().cpu().numpy()[0]
        probability = pred_logits.detach().cpu().numpy()[0,label_id]
        label_name = config_param.relation_id2label[label_id]

        torch.cuda.empty_cache()

        result = {'label_name':label_name,'label_id':label_id,'probability':probability}

        return result
