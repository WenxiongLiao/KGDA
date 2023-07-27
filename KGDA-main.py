import shutil
import os
from  config import config_param
from data_processing import build_remote_corpus
import numpy as np
import pandas as pd
import torch
import os

from data_processing.data_utils import read_common_words,load_pickle,dump_pickle
from data_processing.build_remote_corpus import get_remote_corpus_iter,get_concept_semantic_confidence_AC
from  config import config_param
from model.NER_model import NER
from model.Relation_model import Relation



def init_params():
    config_param.device  = "cuda:2"
    config_param.with_entity_type = True
    config_param.with_cumulative = True
    config_param.need_pred = True
    config_param.model_checkpoint = "./save_models/bert-base-uncased" 
    config_param.iter_N = 6
    config_param.iter_interval = 40000

    config_param.model_name = config_param.model_checkpoint.split('/')[-1]
    config_param.model_name = config_param.model_name + ('_iter' if type(config_param.iter_interval)!= list else '')
    config_param.model_name = config_param.model_name + ('_type' if config_param.with_entity_type == True else '')
    config_param.model_name = config_param.model_name + ('_cumul' if config_param.with_cumulative == True else '')
    config_param.NER_model_save_dir = './save_models/{model_name}_NER'.format(model_name = config_param.model_name)
    config_param.relation_model_save_dir = './save_models/{model_name}_Relation'.format(model_name = config_param.model_name)


    if os.path.exists(config_param.NER_model_save_dir):
        shutil.rmtree(config_param.NER_model_save_dir)
        
    if os.path.exists(config_param.relation_model_save_dir):
        shutil.rmtree(config_param.relation_model_save_dir)

    config_param.remote_corpus_save_paths =  ['./data/remote_corpus_{model_name}_{i}.txt'.format(model_name=config_param.model_name,i = i + 1) for i in range(config_param.iter_N)]


    print(config_param.model_name)



def main():
    # step1 initialize_params
    init_params()

    #step1 load data
    build_remote_corpus.build_remote_corpus()
    remote_corpus_save_paths = config_param.remote_corpus_save_paths
    need_pred = config_param.need_pred

    NER_model = NER()
    relation_model = Relation()

    concept_semantic_AC = load_pickle(config_param.save_concept_semantic_path)  #envir: ['envir', 'CD','CN00079258']
    CID_relation_dict = load_pickle(config_param.save_CID_relation_path) #CN00000003_CN04217333: 1
    new_concept_semantic_dict = {}  #new entities
    new_concept_relation_dict = {}  #new triples
    domain_concept_relation_dict = {}  #overlap triples
    domain_concept_dict = {} #overlap entities


    #step2: train and test on held-out dataset
    for i in range(len(remote_corpus_save_paths)):
        remote_corpus_save_path = remote_corpus_save_paths[i]
        print('process '+ remote_corpus_save_path)
        if i == 0:
    #         The first iteration need_pred is False
            new_concept_relation_dict,domain_concept_relation_dict,domain_concept_dict,remote_corpus_csv_path  = get_remote_corpus_iter(concept_semantic_AC= concept_semantic_AC, 
                            entity_label_word_dict= config_param.entity_label_word_dict,CID_relation_dict = CID_relation_dict,new_concept_relation_dict = new_concept_relation_dict,
                            RELID_name_dict= config_param.RELID_name_dict, relation_label2id =  config_param.relation_label2id,
                            domain_concept_relation_dict=domain_concept_relation_dict,domain_concept_dict=domain_concept_dict,common_words_path=config_param.common_words_path,
                            remote_corpus_save_path=remote_corpus_save_path,need_pred = False,with_cumulative=False)
            NER_model.train_NER_model(remote_corpus_csv_path)
            relation_model.train_relation_model(remote_corpus_csv_path)
            
        elif i == len(remote_corpus_save_paths) - 1:
            #held-out test
            new_concept_relation_dict,domain_concept_relation_dict,domain_concept_dict,remote_corpus_csv_path  = get_remote_corpus_iter(concept_semantic_AC= concept_semantic_AC, 
                            entity_label_word_dict= config_param.entity_label_word_dict,CID_relation_dict = CID_relation_dict,new_concept_relation_dict = new_concept_relation_dict,
                            RELID_name_dict= config_param.RELID_name_dict, relation_label2id =  config_param.relation_label2id,
                            domain_concept_relation_dict=domain_concept_relation_dict,domain_concept_dict=domain_concept_dict,common_words_path=config_param.common_words_path,
                            remote_corpus_save_path=remote_corpus_save_path,need_pred = False,is_test = True,with_cumulative=False)
            NER_model.test_NER_model(remote_corpus_csv_path)
            relation_model.test_relation_model(remote_corpus_csv_path)
            
        else:
            with_cumulative = config_param.with_cumulative
            if need_pred == True:
                #predict entities
                new_concept_semantic_dict = NER_model.NER_pred_file(remote_corpus_save_path,concept_semantic_AC,new_concept_semantic_dict)
                #Find entities with high confidence
                concept_semantic_confidence_AC = get_concept_semantic_confidence_AC(new_concept_semantic_dict,config_param.NER_confidence_num,config_param.NER_confidence_prob)
                
                new_concept_relation_dict,domain_concept_relation_dict,domain_concept_dict,remote_corpus_csv_path  = get_remote_corpus_iter(concept_semantic_AC= concept_semantic_AC, 
                                entity_label_word_dict= config_param.entity_label_word_dict,CID_relation_dict = CID_relation_dict,
                                RELID_name_dict= config_param.RELID_name_dict,relation_label2id =  config_param.relation_label2id,domain_concept_relation_dict=domain_concept_relation_dict,
                                domain_concept_dict=domain_concept_dict,common_words_path=config_param.common_words_path,remote_corpus_save_path=remote_corpus_save_path,
                                relation_model = relation_model,concept_semantic_confidence_AC = concept_semantic_confidence_AC,
                            new_concept_relation_dict = new_concept_relation_dict ,need_pred = need_pred,with_cumulative = with_cumulative)
            else:
                new_concept_relation_dict,domain_concept_relation_dict,domain_concept_dict,remote_corpus_csv_path  = get_remote_corpus_iter(concept_semantic_AC= concept_semantic_AC, 
                            entity_label_word_dict= config_param.entity_label_word_dict,CID_relation_dict = CID_relation_dict,
                            RELID_name_dict= config_param.RELID_name_dict, relation_label2id =  config_param.relation_label2id,
                            domain_concept_relation_dict=domain_concept_relation_dict,domain_concept_dict=domain_concept_dict,common_words_path=config_param.common_words_path,
                            remote_corpus_save_path=remote_corpus_save_path,need_pred = need_pred,with_cumulative=with_cumulative)
                            
            NER_model.train_NER_model(remote_corpus_csv_path)
            relation_model.train_relation_model(remote_corpus_csv_path)


    #step3: save data
    if not os.path.exists('./data/AC_dict/{model_name}'.format(model_name = config_param.model_name)):
        os.makedirs('./data/AC_dict/{model_name}'.format(model_name = config_param.model_name))

    dump_pickle('./data/AC_dict/{model_name}/new_concept_semantic_dict.pkl'.format(model_name = config_param.model_name),new_concept_semantic_dict)
    dump_pickle('./data/AC_dict/{model_name}/new_concept_relation_dict.pkl'.format(model_name = config_param.model_name),new_concept_relation_dict)
    dump_pickle('./data/AC_dict/{model_name}/domain_concept_relation_dict.pkl'.format(model_name = config_param.model_name),domain_concept_relation_dict)
    dump_pickle('./data/AC_dict/{model_name}/domain_concept_dict.pkl'.format(model_name = config_param.model_name),domain_concept_dict)
    dump_pickle('./data/AC_dict/{model_name}/concept_semantic_confidence_AC.pkl'.format(model_name = config_param.model_name),concept_semantic_confidence_AC)


    new_concept_relation_confidence_dict = {}
    for k,v in new_concept_relation_dict.items():
        if len(v[1])>=config_param.relation_confidence_num and np.mean(v[1])>config_param.relation_confidence_prob:
            new_concept_relation_confidence_dict[k] = v[0]
    dump_pickle('./data/AC_dict/{model_name}/new_concept_relation_confidence_dict.pkl'.format(model_name = config_param.model_name),new_concept_relation_confidence_dict)


if __name__== "__main__" :
    main()

