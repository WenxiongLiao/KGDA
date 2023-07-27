import ahocorasick
import json
import numpy as np
import re
import pandas as pd
from pip import main
from torch import true_divide
from tqdm.auto import tqdm
import itertools
from nltk.stem.porter import * 

from data_processing.data_washing import process_raw_corpus,drop_duplicates_lines
from data_processing.data_utils import *
from  config import config_param




def build_concept_AC(conceptTerms_path):
    # Read conceptTerms and construct conceptTerm_AC
    conceptTerms = read_txt(conceptTerms_path)
    conceptTerm_dict = {}
    CID_concept_dict = {}
    for term in conceptTerms:
        CID, SID, STR, LANG, TTY = term.split('|')
        if LANG == 'ENG':
            conceptTerm_dict[STR.lower()] = [CID,STR.lower()]
            CID_concept_dict[CID] = STR.lower()
    print('{num} concept term'.format(num = len(conceptTerm_dict)))
    conceptTerm_AC =  build_ahocorasick(conceptTerm_dict)
    # CID_concept_AC = build_ahocorasick(CID_concept_dict)

    return conceptTerm_AC

def get_trip_before_end(text,end):
    t = text[:end + 1].split()
    end_t = t[-1]
    if end - len(end_t) <= 0:
        return 0
    else:
        return end - len(end_t)
    

def get_trip_after_end(text,end):
    t = text[end:].split()
    end_t = t[0]
    return end + len(end_t) 

def get_line_remote_entity(line,concept_semantic_AC,concept_semantic_confidence_AC,common_words,need_pred = False):
    if need_pred:
        assert concept_semantic_confidence_AC != None, print('params error')
    porter_stemmer = PorterStemmer() 
    line = line.strip()
    mentions = list(concept_semantic_AC.iter(line))
    if need_pred  and len(concept_semantic_confidence_AC) > 0:
        #print(len(concept_semantic_confidence_AC))
        mentions2 = list(concept_semantic_confidence_AC.iter(line))
        mentions =  mentions2 + mentions 
    
    # we need to filter mentions
    start_end_idx = []
    mentions_drop_inclue = []
    # (202, ['colloid','CD',CID]),
    #  (207, ['colloid cyst','CD',CID]),
    #  (208, ['colloid cysts','CD',CID]),
    #  only retain ((208, ['colloid cysts','CD',CID])  if from concept_semantic_confidence_AC  CID = 'NER_prediction'
    for mention in list(reversed(mentions)):
        if len(start_end_idx)==0:
            start = mention[0] - len(mention[1][0]) + 1
            end = mention[0]
            (real_start,real_end) = (start,end)
            trip_before_end = get_trip_before_end(line,end)
            trip_after_end = get_trip_after_end(line,end) - 1
            
            term_stem = porter_stemmer.stem(line[start : trip_after_end + 1].strip())
            term = line[start : trip_after_end + 1].strip()
            if concept_semantic_confidence_AC != None:
                if  (term_stem in concept_semantic_AC or term in concept_semantic_AC or term_stem in concept_semantic_confidence_AC or term in concept_semantic_confidence_AC ) and term_stem not in  common_words  and term not in  common_words:
                    start_end_idx = [start,trip_after_end]
                    mentions_drop_inclue.append([start_end_idx[0],start_end_idx[1],line[start_end_idx[0] : start_end_idx[1] + 1],mention[1][1],mention[1][2] ])
            else:
                if  (term_stem in concept_semantic_AC or term in concept_semantic_AC ) and term_stem not in  common_words  and term not in  common_words:
                    start_end_idx = [start,trip_after_end]
                    mentions_drop_inclue.append([start_end_idx[0],start_end_idx[1],line[start_end_idx[0] : start_end_idx[1] + 1],mention[1][1],mention[1][2] ])
        else:
            start = mention[0] - len(mention[1][0]) + 1
            end = mention[0] 
            (real_start,real_end) = (start,end)
            if start>=start_end_idx[0] and end<=start_end_idx[1]:
                #include
                pass
            else:
                trip_before_end = get_trip_before_end(line,end)
                trip_after_end = get_trip_after_end(line,end) - 1
                
                term_stem = porter_stemmer.stem(line[start : trip_after_end + 1].strip())
                term = line[start : trip_after_end + 1].strip()
                if concept_semantic_confidence_AC != None:
                    if  (term_stem in concept_semantic_AC or term in concept_semantic_AC or term_stem in concept_semantic_confidence_AC or term in concept_semantic_confidence_AC ) and term_stem not in  common_words  and term not in  common_words:
                        start_end_idx = [start,trip_after_end]
                        mentions_drop_inclue.append([start_end_idx[0],start_end_idx[1],line[start_end_idx[0] : start_end_idx[1] + 1],mention[1][1],mention[1][2] ])
                else:
                    if  (term_stem in concept_semantic_AC or term in concept_semantic_AC ) and term_stem not in  common_words  and term not in  common_words:
                        start_end_idx = [start,trip_after_end]
                        mentions_drop_inclue.append([start_end_idx[0],start_end_idx[1],line[start_end_idx[0] : start_end_idx[1] + 1],mention[1][1],mention[1][2] ])
   
            
    end_idx = []
    mentions_drop_dup = []
    #   [163, 168, 'carbon','CD',CID],
    #  [159, 168, 'ydrocarbon','CD',CID],
    #  [158, 168, 'hydrocarbon','CD'],
    #  [145, 168, 'carcinogenic hydrocarbon','CD',CID],
    #  only retain [145, 168, 'carcinogenic hydrocarbon','CD',CID]   if from concept_semantic_confidence_AC  CID = 'NER_prediction'
    for mention in list(reversed(mentions_drop_inclue)):
        if mention[1] in end_idx:
            pass
        else:
            if line[mention[1] - len(mention[2])] == ' ' or mention[1] - len(mention[2]) < 0:
                if len(line)>= mention[1] + 1 and   line[mention[1] + 1 ]  not in ['s','e',' ',',' ,'.','?','*'] :
                    pass
                else:
                    concept_len = len(line[mention[0]:mention[1] + 1].split())
                    begin_token_idx =  len(line[0:mention[0]].split())
                    end_token_idx = begin_token_idx + concept_len
                    mention.append(begin_token_idx)
                    mention.append(end_token_idx)
                    mentions_drop_dup.append(mention)
                    end_idx.append(mention[1])

    mentions = mentions_drop_dup
#     mention = [begin_idx, end_idx, conceptTerm, entity_label_word, CID, begin_token_idx, end_token_idx]  if from concept_semantic_confidence_AC  CID = 'NER_prediction'
    return mentions

def get_entity_labels(line, mentions, entity_label_word_dict):
    tags = ['O'] * len(line.strip().split())

    for term in mentions:
        [begin_idx, end_idx, conceptTerm,entity_label,CID, begin_token_idx, end_token_idx] = term
        
        for i in range(begin_token_idx, end_token_idx):
            if i == begin_token_idx:
                tags[i] = 'B-' + entity_label
            else:
                tags[i] = 'I-' + entity_label
    return tags

def get_combine_mentions_relation(combine_mentions,sentence,relation_model,label_word_entity_dict):
#       mention = [begin_idx, end_idx, conceptTerm, entity_label_word, CID, begin_token_idx, end_token_idx]  if from concept_semantic_confidence_AC  CID = 'NER_prediction'
    head_entity_list = [combine[0][2] for combine in combine_mentions]
    tail_entity_list = [combine[1][2] for combine in combine_mentions]
    head_entity_type_list = [combine[0][3] for combine in combine_mentions]
    head_entity_type_list = [label_word_entity_dict[entity_type] for entity_type in head_entity_type_list]
    tail_entity_type_list = [combine[1][3] for combine in combine_mentions]
    tail_entity_type_list = [label_word_entity_dict[entity_type] for entity_type in tail_entity_type_list]

    sentences = [sentence] * len(tail_entity_list)
    
    if config_param.with_entity_type == False:
        results = relation_model.relation_pred_batch(head_entity_list,tail_entity_list,sentences)
    else:
        results = relation_model.relation_pred_batch(head_entity_list,tail_entity_list,sentences,head_entity_type_list,tail_entity_type_list)
    
    return results


def get_remote_corpus_iter(concept_semantic_AC, entity_label_word_dict,CID_relation_dict,RELID_name_dict,relation_label2id,
                        domain_concept_relation_dict,domain_concept_dict,common_words_path,remote_corpus_save_path
                        ,relation_model = None,concept_semantic_confidence_AC = None,
                       new_concept_relation_dict = None,need_pred = False,with_cumulative = False,is_test = False):
    if need_pred:
        #If the prediction results of NER model and relation model are required
        assert relation_model!= None and concept_semantic_confidence_AC!= None and new_concept_relation_dict!=None  , print("params error!")

    relation_label_dict = relation_label2id
    porter_stemmer = PorterStemmer() 
    label_word_entity_dict = {v: k for k, v in entity_label_word_dict.items()}
    
    filter_lines = read_txt(remote_corpus_save_path)
    
    common_words = read_common_words(common_words_path)

    tokens_list = []
    tags_list = []
    relations_list = []
    for line in tqdm(filter_lines):
        tokens = line.strip().split()
        #Find the entities mentioned in the sentence
        mentions = get_line_remote_entity(line,concept_semantic_AC,concept_semantic_confidence_AC,common_words,need_pred)
        if is_test == False:
            for mention in mentions:
                domain_concept_dict[mention[2]] = [mention[2],mention[3],mention[4]] 
        tags = get_entity_labels(line, mentions,  entity_label_word_dict)
        #Combine the entities in pairs to find the relationship between pairs
        combine_mentions = list(itertools.permutations(mentions, 2))
        if need_pred:
            relation_pred = get_combine_mentions_relation(combine_mentions,line,relation_model,label_word_entity_dict)
        
        relations = []
        relations_set = set()
#       mention = [begin_idx, end_idx, conceptTerm, entity_label_word, CID, begin_token_idx, end_token_idx]  if from concept_semantic_confidence_AC  CID = 'NER_prediction'
        for i in range(len(combine_mentions)):
            combine = combine_mentions[i]
            
            if combine[0][2] !=combine[1][2]:
                is_existence = False
                if combine[0][4] != 'NER_prediction' and  combine[1][4] != 'NER_prediction' and combine[0][4] + '_' + combine[1][4] in CID_relation_dict.keys():
                    #if exist in CID_relation_dict
                    relation_type = relation_label2id[ RELID_name_dict[ CID_relation_dict[combine[0][4] + '_' + combine[1][4]]] ]
                    if is_test == False:
                        domain_concept_relation_dict[combine[0][4] + '_' + combine[1][4]] = relation_type
                    if combine[0][2] + '_' + combine[1][2] not in relations_set:
                        relations.append({'head':combine[0][2], 'tail':combine[1][2],'head_type':label_word_entity_dict[combine[0][3]], 'tail_type':label_word_entity_dict[combine[1][3]],'head_idx':combine[0][5:], 'tail_idx':combine[1][5:], 'relation': relation_type})
                        relations_set.add(combine[0][2] + '_' + combine[1][2])
                    is_existence = True
                else:
                    if need_pred:
                        result = {'label_name': relation_pred['label_name'][i],'label_id':int(relation_pred['label_id'][i]),'probability':relation_pred['probability'][i]}
                        if result['label_id']!=relation_label_dict['NULL']:
                            # combine_concept = porter_stemmer.stem(combine[0][2].strip()) + '_|_' + porter_stemmer.stem(combine[1][2].strip())
                            combine_concept = combine[0][2].strip()+ '_|_' + combine[1][2].strip()
                            if combine_concept in new_concept_relation_dict.keys():
                                #if exist in new_concept_relation_dict
                                is_existence = True
                                [relation_type,probs] = new_concept_relation_dict[combine_concept]
                                if relation_type == result['label_id']:
                                    probs = list(set(probs + [result['probability']])) #Use sets to deduplicate, because some sentences appear multiple times with the same entity
                                    new_concept_relation_dict[combine_concept] = [relation_type,probs[-10:] ]   # only the last ten probs
                                else:
                                    new_concept_relation_dict[combine_concept] = [result['label_id'],[result['probability']] ]
                                [relation_type,probs] = new_concept_relation_dict[combine_concept]
                                if len(probs)> config_param.relation_confidence_num and np.mean(probs)> config_param.relation_confidence_prob:
                                    if combine[0][2] + '_' + combine[1][2] not in relations_set:
                                        relations.append({'head':combine[0][2], 'tail':combine[1][2],'head_type':label_word_entity_dict[combine[0][3]], 'tail_type':label_word_entity_dict[combine[1][3]],'head_idx':combine[0][5:], 'tail_idx':combine[1][5:], 'relation': relation_type})  
                                        relations_set.add(combine[0][2] + '_' + combine[1][2])

                            if is_existence==False:
                                #If it does not exist in concept_relation_dict, nor in new_concept_relation_dict, you need to input the relation model prediction
                                new_concept_relation_dict[combine_concept] = [result['label_id'],[result['probability']] ]

        tokens_list.append(tokens)
        tags_list.append(tags)
        relations_list.append(relations)

    remote_corpus = pd.DataFrame({'tokens':tokens_list,'tags':tags_list,'relations':relations_list})
    [path ,suffix] = remote_corpus_save_path.split('.') if remote_corpus_save_path.startswith('.') == False else remote_corpus_save_path[1:].split('.')
    remote_corpus_csv_path = ('.' if remote_corpus_save_path.startswith('.') == True else '')  + path  + '.' + 'csv'
    if with_cumulative:
        file_idx = eval(path.split('_')[-1]) - 1
        cumulative_data_path = ('.' if remote_corpus_save_path.startswith('.') == True else '')  + path[:-len(str(file_idx))] + str(file_idx) + '.' + 'csv'
        cumulative_df = pd.read_csv(cumulative_data_path)
        remote_corpus = pd.concat([cumulative_df,remote_corpus])
    remote_corpus.to_csv(remote_corpus_csv_path,index = False)

    return new_concept_relation_dict,domain_concept_relation_dict,domain_concept_dict,remote_corpus_csv_path


def get_concept_semantic_confidence_AC(new_concept_semantic_dict,NER_confidence_num = 2,NER_confidence_prob = 0.98):
    concept_semantic_dict_confidence = {}
#     new_concept_semantic_dict[result['word']] = (result['word'],semantic,'NER_prediction',[result['score']])
    for k,v in new_concept_semantic_dict.items():
        confidence_list = v[3]
        if len(confidence_list) >= NER_confidence_num and np.mean(confidence_list)>=NER_confidence_prob:
            concept_semantic_dict_confidence[k] = [k,v[1],v[2]]
            
    concept_semantic_confidence_AC = build_ahocorasick(concept_semantic_dict_confidence)
    
    return concept_semantic_confidence_AC



def build_term_entity_AC(semanticTypes_path):
    semanticTypes_lines = read_txt(semanticTypes_path)
    term_entity_dict = {}
    for term in semanticTypes_lines:
        CID, STYID, STY = term.split('|')
        term_entity_dict[CID] = STY.strip()
        
    print('{num} semanticTypes term'.format(num = len(term_entity_dict)))
    term_entity_AC =  build_ahocorasick(term_entity_dict)

    return  term_entity_AC




def build_CID_relation_dict(relations_path):
    relations_lines = read_txt(relations_path)

    CID_relation_dict = {}
    for line in relations_lines:
        RID, CID_HEAD, CID_TAIL, RELID, REL = line.split('|')
        CID_relation_dict[CID_HEAD + '_' + CID_TAIL] = eval(RELID)
        
    print('{num} relations term'.format(num = len(CID_relation_dict)))
    # CID_relation_AC =  build_ahocorasick(CID_relation_dict)
    return CID_relation_dict

def  build_RELID_name_AC(relations_name_path):
    relations_lines = read_txt(relations_name_path)

    RELID_name_dict = {}
    for line in relations_lines:
        RELID,LANG, REL = line.split('|')
        if LANG == 'ENG':
            RELID_name_dict[RELID] = REL.strip()
        
    print('{num} relation names'.format(num = len(RELID_name_dict)))
    RELID_name_AC =  build_ahocorasick(RELID_name_dict)

    return RELID_name_AC



def split_remote_corpus(raw_corpus_filter_path,remote_corpus_save_paths):
    lines = read_txt(raw_corpus_filter_path)
    
    if type(config_param.iter_interval) == list:
        assert len(config_param.iter_interval) == config_param.iter_N + 1,print('params error')
        split_idx = config_param.iter_interval
    else:
        split_idx = list(range(0,(config_param.iter_N + 1) * config_param.iter_interval ,config_param.iter_interval))

    print(split_idx)
    # split_idx = [0,40000,80000,120000,160000,200000]


    for i in range(len(split_idx)):
        remote_corpus_save_path = remote_corpus_save_paths[i-1]
        if i == 0:
            pass
        else:
            remote_corpus_split = lines[split_idx[i - 1] : split_idx[i]]
            print(len(remote_corpus_split))
            [path ,suffix] = remote_corpus_save_path.split('.') if remote_corpus_save_path.startswith('.') == False else remote_corpus_save_path[1:].split('.')
            write_line_txt(remote_corpus_split,('.' if remote_corpus_save_path.startswith('.') == True else '') + path + '.' + suffix)


def save_dict(conceptTerm_AC,term_entity_AC,CID_relation_dict,save_concept_semantic_path,save_CID_relation_path):
    concept_semantic_dict = {}
    for k,v in conceptTerm_AC.items():
        concept_semantic_dict[k] = [k,config_param.entity_label_word_dict[ term_entity_AC.get(v[0]) ] ,v[0] ] #lung cancer: [lung cancer,NP,CN0000xxx]

    concept_semantic_AC =  build_ahocorasick(concept_semantic_dict)
    dump_pickle(save_concept_semantic_path,concept_semantic_AC)
    dump_pickle(save_CID_relation_path,CID_relation_dict)



def build_remote_corpus():

    #step0 set the path and parameter
    conceptTerms_path = config_param.conceptTerms_path
    relations_path = config_param.relations_path
    semanticTypes_path = config_param.semanticTypes_path
    oncology_corpus_path = config_param.oncology_corpus_path
    oncology_corpus_filter_path = config_param.oncology_corpus_filter_path
    remote_corpus_save_paths = config_param.remote_corpus_save_paths
    save_concept_semantic_path = config_param.save_concept_semantic_path
    save_CID_relation_path = config_param.save_CID_relation_path


    # #step1 process_raw_corpus  
    process_raw_corpus(oncology_corpus_path,oncology_corpus_filter_path)  #Please skip this step, the original dataset will be released next month
    drop_duplicates_lines(oncology_corpus_filter_path)

    # step2 build AC
    conceptTerm_AC = build_concept_AC(conceptTerms_path)
    term_entity_AC = build_term_entity_AC(semanticTypes_path)
    CID_relation_dict = build_CID_relation_dict(relations_path)

    # #step 3  save_dict
    save_dict(conceptTerm_AC,term_entity_AC,CID_relation_dict,save_concept_semantic_path,save_CID_relation_path)

    # # step 4  split_remote_corpus
    split_remote_corpus(oncology_corpus_filter_path,remote_corpus_save_paths) # radio




if __name__ == '__main__':
    build_remote_corpus()
