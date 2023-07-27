import langid
from data_processing.data_utils import build_ahocorasick
# import fasttext
# from transformers import pipeline

# model = fasttext.load_model('./data/lid.176.ftz')
# lang_cls = pipeline("text-classification", model = "papluca/xlm-roberta-base-language-detection",device = 3)

def check_end(entity):
    '''
    Filter out incomplete entities
    '''
    words = entity.split()
    if len(words) > 1 :
        if words[-1].strip() in ['is','are','of','for','by','with','vs','between','at','in','on']:
            return False
    return True

def is_english(word,corpus_lines):
#     if langid.classify(word)[0] == 'en' or model.predict(word, k=1)[0][0].split('__')[-1] == 'en' or lang_cls(word)[0]['label'] == 'en':
#         return True
    for line in corpus_lines:
        if word in line:
            if langid.classify(line)[0] == 'en':
                return True
            else:
                return False
    print('miss')
    return False

def check_relation_en(relation_dict,corpus_lines):
    post_relation_dict = {}
    for k,v in relation_dict.items():
        head,tail = k.split('_|_')
        if is_english(head,corpus_lines) and is_english(tail,corpus_lines):
            if check_end(head) and check_end(tail):
                post_relation_dict[k] = v
    
    print('before: {num}, after: {post_num}'.format(num = len(relation_dict),post_num = len(post_relation_dict)))
    
    return post_relation_dict

def check_AC_en(source_AC,corpus_lines):
    post_dict = {}
    for k, v in source_AC.items():
        if is_english(k,corpus_lines) == True:
            if check_end(k):
                post_dict[k] = v
    post_AC = build_ahocorasick(post_dict)
    print('before: {num}, after: {post_num}'.format(num = len(source_AC),post_num = len(post_AC)) )
    

    return post_AC


