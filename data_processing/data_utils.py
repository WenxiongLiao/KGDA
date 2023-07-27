import ahocorasick
import pandas as pd
import pickle
import random
import numpy as np
import torch
import os
import uuid


def load_pickle(variable_path):
    '''
    加载通过pickle持久化的变量
    variable_path:变量保存路径
    '''
    f=open(variable_path,'rb')
    variable = pickle.load(f)
    f.close()
    return variable

def dump_pickle(variable_path,variable):
    '''
    将变量持久化
    variable_path:变量保存路径
    variable:待保存变量
    '''
    f=open(variable_path,'wb')
    pickle.dump(variable,f)
    f.close()

def read_txt(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    return data

def write_line_txt(lines,file_apth):
    with open(file_apth, 'w', encoding= 'utf-8') as f:
        for line in lines:
            f.write(line)
            
def read_common_words(common_words_path, add_suffixes = True):
    common_words_pd = pd.read_csv(common_words_path)
    common_words = []
    for word in common_words_pd.common_words.values:
        word = word.strip().lower()
        if len(word) > 0:
            common_words.append(word)
            if add_suffixes:
                # Processing suffixes
                if word.endswith('ed') or word.endswith('ly')  or word.endswith('es'):
                    common_words.append(word[:-2])
                elif  word.endswith('ing'):
                    common_words.append(word[:-3])
                elif word.endswith('s'):
                    common_words.append(word[:-1])
                else:
                    common_words.append(word + 'ed')
                    common_words.append(word + 'ly')
                    common_words.append(word + 'es')
                    common_words.append(word + 'ing')
                    common_words.append(word + 's')

    return common_words

            
def build_ahocorasick(AC_dict):
    A = ahocorasick.Automaton()
    for key,value in AC_dict.items():
        A.add_word(key, value)
    A.make_automaton()
    return A


def seed_everything(seed_value):

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_sample_id():

    return uuid.uuid1()