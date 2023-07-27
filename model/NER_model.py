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
from transformers import AutoTokenizer,BertTokenizer, BertModel,AutoModel,BertForMaskedLM,AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import get_scheduler
from transformers import pipeline
from datasets import load_metric
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import * 


from data_processing.data_utils import read_common_words,read_txt
from  config import config_param

def global_init():
    global device,tokenizer,metric,data_collator,common_words
    device = torch.device(config_param.device) if torch.cuda.is_available() else torch.device("cpu")
    if Path(config_param.NER_model_save_dir).exists():
        tokenizer = AutoTokenizer.from_pretrained(config_param.NER_model_save_dir, padding=True, truncation=True,model_max_length = 510)#,add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_param.model_checkpoint, padding=True, truncation=True,model_max_length = 510)#,add_prefix_space=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    common_words = read_common_words(config_param.common_words_path)
    metric = load_metric("./metrics/seqeval")


def str_to_list(examples):
    '''
    Convert characters in csv to list
    '''

    all_tokens = examples["tokens"]
    all_tags = examples["tags"]

    new_tokens = []
    new_tags = []
    
    for i, tokens in enumerate(all_tokens):
        new_tokens.append(eval(tokens))
    
    for i, tags in enumerate(all_tags):
        new_tags.append([config_param.entity_label2id[tag] for tag in eval(tags)])
        
    new_examples = {}
    new_examples["tokens"] = new_tokens
    new_examples["tags"] = new_tags

    return new_examples

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
#             new_labels.append(entity_label_dict[label])
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
#             label = entity_label_dict[label]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True,max_length = 510
    )
    all_labels = examples["tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[config_param.entity_label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [config_param.entity_label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions
    
class NER():
    def __init__(self) -> None:
        global_init()
        self.token_classifier = None

    def NER_model_init(self):
        if Path(config_param.NER_model_save_dir).exists():
            self.model = AutoModelForTokenClassification.from_pretrained(
                    config_param.NER_model_save_dir,
                    id2label=config_param.entity_id2label,
                    label2id=config_param.entity_label2id,
                )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                    config_param.model_checkpoint,
                    id2label=config_param.entity_id2label,
                    label2id=config_param.entity_label2id,
                )
        #self.model = nn.DataParallel(self.model,device_ids=[0,1,2])
        self.model.to(device)
        
        
    def get_train_val_dataLoader(self,data_path):
        '''
        load single csv file and split as train_dataloader , eval_dataloader
        '''
        raw_datasets = load_dataset("csv", data_files=data_path)
        raw_datasets = raw_datasets['train'].train_test_split(test_size=config_param.NER_test_ratio, shuffle=True, seed=2022)

        raw_datasets = raw_datasets.map(
            str_to_list,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

        train_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=config_param.NER_batch_size
        )
        eval_dataloader = DataLoader(
            tokenized_datasets["test"], collate_fn=data_collator, batch_size=config_param.NER_batch_size
        )

        return train_dataloader,eval_dataloader
    
    def get_test_dataLoader(self,data_path):
        '''
        load single csv file and split as test_dataLoader
        '''
        raw_datasets = load_dataset("csv", data_files=data_path)

        raw_datasets = raw_datasets.map(
            str_to_list,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

        tokenized_datasets = raw_datasets.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
        )

        test_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=config_param.NER_batch_size
        )


        return test_dataloader

        

    def train_NER_model(self,data_path):
        # init NER_model
        print('model init...')
        self.NER_model_init()

        #get dataLoader
        print('get dataLoader...')
        train_dataloader,eval_dataloader = self.get_train_val_dataLoader(data_path)

        optimizer = AdamW(self.model.parameters(), lr=config_param.NER_lr)

        # init lr
        num_update_steps_per_epoch = len(train_dataloader)
        num_training_steps = config_param.NER_num_train_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        progress_bar = tqdm(range(num_training_steps))

        max_acc = 0
        print('model training...')
        for epoch in range(config_param.NER_num_train_epochs):
            # Training
            self.model.train()
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                
                loss.backward()
                #loss.sum().backward()
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                progress_bar.update(1)

            # Evaluation
            self.model.eval()
            for batch in eval_dataloader:
                with torch.no_grad():
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = self.model(**batch)

                predictions = outputs.logits.argmax(dim=-1)
                labels = batch["labels"]

                true_predictions, true_labels = postprocess(predictions, labels)
                metric.add_batch(predictions=true_predictions, references=true_labels)

            results = metric.compute()
            print(
                f"epoch {epoch}:",
                {
                    key: results[f"overall_{key}"]
                    for key in ["precision", "recall", "f1", "accuracy"]
                },
            )

            if results[f"overall_accuracy"] > max_acc:
                max_acc = results[f"overall_accuracy"]
                print('save model')
                self.model.save_pretrained(config_param.NER_model_save_dir)
                #self.model.module.save_pretrained(config_param.NER_model_save_dir)
                
        tokenizer.save_pretrained(config_param.NER_model_save_dir)
        self.token_classifier = None
        
        torch.cuda.empty_cache()

    
    def test_NER_model(self,data_path):
        # init NER_model
        print('model init...')
        self.NER_model_init()

        #get dataLoader
        print('get dataLoader...')
        test_dataloader = self.get_test_dataLoader(data_path)
        self.model.eval()
        for batch in tqdm(test_dataloader):
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)

            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]

            true_predictions, true_labels = postprocess(predictions, labels)
            metric.add_batch(predictions=true_predictions, references=true_labels)

        results = metric.compute()
        print({
                key: results[f"overall_{key}"]
                for key in ["precision", "recall", "f1", "accuracy"]
            },)
        torch.cuda.empty_cache()


    def NER_pred_file(self,pred_file_path,concept_semantic_AC,new_concept_semantic_dict):
       
        porter_stemmer = PorterStemmer() 
        pred_sentences = read_txt(pred_file_path)
        for pred_sentence in tqdm(pred_sentences):
            pred_results = self.NER_pred(pred_sentence)

            for result in pred_results:
                if result['word'] in concept_semantic_AC or (result['word'].strip().endswith('s') and  result['word'].strip()[:-1] in concept_semantic_AC) or  porter_stemmer.stem(result['word'].strip()) in concept_semantic_AC:
                    # If result['word'] in concept_semantic_AC or the singular form of result['word'] exists in concept_semantic_AC
                    pass
                else:
                    if result['word'] in new_concept_semantic_dict.keys():
                        (concept,semantic,CID,probs) =  new_concept_semantic_dict[result['word']]
                        if semantic == result['entity_group']:
                            probs = probs + [result['score']]
                            new_concept_semantic_dict[result['word']] = (result['word'],semantic,CID,probs[-10:] ) # only the last ten probs
                    else:
                        new_concept_semantic_dict[result['word']] = (result['word'],result['entity_group'],'NER_prediction',[result['score']])
        
        return new_concept_semantic_dict
                    

    def NER_pred(self,sentence):
        if self.token_classifier == None:
 
            tokenizer = AutoTokenizer.from_pretrained(config_param.NER_model_save_dir, padding=True, truncation=True,model_max_length = 510)#,add_prefix_space=True)
            self.token_classifier = pipeline(
                "ner", model = config_param.NER_model_save_dir, aggregation_strategy="simple",tokenizer = tokenizer,device = 0
            )


        results = self.token_classifier(sentence.strip().lower())
        new_results = []
        for result in results:
            if result['word'] not in common_words and len(result['word'])>=3 and  result['word'].isdigit() == False:
                if (result['start'] == 0 or sentence[result['start'] - 1] == ' ') and (result['end'] == len(sentence) or sentence[result['end']] in [' ','!','?','C',';',',','.']):
                    result['word'] = result['word'].strip()
                    new_results.append(result)

        return new_results


