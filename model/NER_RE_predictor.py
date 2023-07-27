import numpy as np
import itertools


def NER_RE_pred(paragraph,NER_model,relation_model,entity_label_word_dict,relation_id2label,with_entity_type = True,NER_confidence_prob = 0.95,relation_confidence_prob = 0.97):
    '''
    Extract entities from a paragraph and identify their relationships
    '''
    label_word_entity_dict = {v: k for k, v in entity_label_word_dict.items()}

    NER_result = NER_model.NER_pred(paragraph)
    mentions = []
    for result in NER_result:
        if result['score'] > NER_confidence_prob:
            mentions.append([result['word'],label_word_entity_dict[result['entity_group']]])
    combine_mentions = list(itertools.permutations(mentions, 2))

    head_tail_set = set()

    head_entity_list = []
    tail_entity_list = []
    head_entity_type_list = []
    tail_entity_type_list = []
    for i in range(len(combine_mentions)):
        combine = combine_mentions[i]
        head_entity = combine[0][0]
        tail_entity = combine[1][0]
        if head_entity + '_' + tail_entity not in head_tail_set:
            head_tail_set.add(head_entity + '_' + tail_entity)
            if head_entity != tail_entity:
                head_entity_list.append(head_entity)
                tail_entity_list.append(tail_entity)
                if with_entity_type:
                    head_entity_type_list.append(combine[0][1])
                    tail_entity_type_list.append(combine[1][1])
        else:
            continue

    paragraph = [paragraph] * len(tail_entity_list)            
    if with_entity_type:
        result = relation_model.relation_pred_batch(head_entity_list,tail_entity_list,paragraph,head_entity_type_list,tail_entity_type_list)
    else:
        result = relation_model.relation_pred_batch(head_entity_list,tail_entity_list,paragraph)
    
    relation_names = [relation_id2label[ids] for ids in result['label_id']]
    result['relation_names'] = relation_names

    select_idx = np.array(list(range(len(result['label_name']))))[np.logical_and(np.array(result['probability'])> relation_confidence_prob, np.array(result['relation_names'])!= 'NULL')] 
    
    if len(select_idx)>0:
        result = {'head_entity':np.array(head_entity_list)[select_idx],'tail_entity':np.array(tail_entity_list)[select_idx],'relation_names':np.array(result['relation_names'])[select_idx],'probability':np.array(result['probability'])[select_idx]}

        result = [[head,tail,relation,probability]  for head,tail,relation,probability in zip(result['head_entity'],result['tail_entity'],result['relation_names'],result['probability'])]
    
        return result
    else:
        return []



