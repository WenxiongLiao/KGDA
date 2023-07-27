
#data save path
conceptTerms_path = './data/BIOS 2022v1(beta)/CoreData/ConceptTerms.txt'  # download BIOS 2022v1(beta) from https://bios.idea.edu.cn/Download
definitions_path = './data/BIOS 2022v1(beta)/CoreData/Definitions.txt'
relations_path = './data/BIOS 2022v1(beta)/CoreData/Relations.txt'
relations_name_path = './data/BIOS 2022v1(beta)/CoreData/RelationNames.txt'
semanticTypes_path = './data/BIOS 2022v1(beta)/CoreData/SemanticTypes.txt'
oncology_corpus_path = './data/Copy_of_all_clean_formatted.txt'   # This original dataset will be released next month
oncology_corpus_filter_path = './data/Copy_of_all_clean_formatted_filter.txt' # Dataset after preprocessing

common_words_path = './data/common_words.csv'  # out-of-domain words

save_concept_semantic_path = './data/AC_dict/concept_semantic_AC.pkl'  # The concept AC of BIOS KG
save_CID_relation_path = './data/AC_dict/CID_relation_dict.pkl'  # The relationships of BIOS KG

device  = "cuda:0"
with_entity_type = True
with_cumulative = False
need_pred = True
# model_checkpoint = "./save_models/emilyalsentzer-Bio_ClinicalBERT"  # download from https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
# model_checkpoint = "./save_models/allenai-biomed_roberta_base"  # download from https://huggingface.co/allenai/biomed_roberta_base
model_checkpoint = "./save_models/bert-base-uncased"          #download from  https://huggingface.co/bert-base-uncased

iter_N = 6
iter_interval = 40000

remote_corpus_save_path = './data/remote_corpus.txt'
remote_corpus_save_paths =  ['./data/remote_corpus{i}.txt'.format(i = i + 1) for i in range(iter_N)]
# remote_corpus_save_paths = ['./data/remote_corpus1.txt','./data/remote_corpus2.txt','./data/remote_corpus3.txt','./data/remote_corpus4.txt','./data/remote_corpus5.txt']

model_name = model_checkpoint.split('/')[-1]
model_name = model_name + ('_iter' if type(iter_interval)!= list else '')
model_name = model_name + ('_type' if with_entity_type == True else '')
NER_model_save_dir = './save_models/{model_name}_NER'.format(model_name = model_name)
relation_model_save_dir = './save_models/{model_name}_Relation'.format(model_name = model_name)



#NER params
entity_label_word_dict = {'Anatomy':'Anat','Sign, Symptom, or Finding':'SSF','Neoplastic Process':'NP',
                       'Microorganism':'Micro', 'Eukaryote':'Eukar','Physiology':'Physi','Chemical or Drug':'CD',
                        'Diagnostic Procedure':'DP','Laboratory Procedure':'LP','Research Activity or Technique':'RAT',
                          'Therapeutic or Preventive Procedure':'TP','Medical Device':'MD','Research Device':'RD',
                          'Pathology':'Patho','Disease or Syndrome':'DS','Anatomical Abnormality':'AA','Mental or Behavioral Dysfunction':'MBD',
                          'Injury or Poisoning':'IP'
}
label_word_entity_dict = {v: k for k, v in entity_label_word_dict.items()}

entity_label_names = ['O', 'B-Anat', 'I-Anat', 'B-SSF', 'I-SSF', 'B-NP', 'I-NP', 'B-Micro', 'I-Micro', 'B-Eukar', 'I-Eukar', 'B-Physi', 'I-Physi', 'B-CD', 'I-CD', 'B-DP', 'I-DP', 'B-LP', 'I-LP', 'B-RAT', 'I-RAT','B-TP', 'I-TP', 'B-MD', 'I-MD', 'B-RD', 'I-RD', 'B-Patho', 'I-Patho', 'B-DS', 'I-DS', 'B-AA', 'I-AA', 'B-MBD', 'I-MBD', 'B-IP', 'I-IP']
entity_id2label = {i: label for i, label in enumerate(entity_label_names)}
entity_label2id = {v: k for k, v in entity_id2label.items()}


# RELID in the relations_name_path
RELID_name_dict = {1: 'is_a', 2: 'reverse_is_a', 3: 'is_part_of', 4: 'reverse_is_part_of', 5: 'may_treat', 6: 'reverse_may_treat', 9: 'found_in', 10: 'reverse_found_in', 
                  11: 'may_cause', 12: 'reverse_may_cause', 13: 'expressed_in', 14: 'is_expression_of', 15: 'encodes', 16: 'encoded_by', 17: 'significant_drug_interaction', 
                  34: 'involved_in_biological_process', 36: 'biological_process_involves', 39: 'is_active_ingredient_in', 42: 'has_active_ingredient'}

NER_confidence_num = 2
NER_confidence_prob = 0.95
NER_test_ratio = 0.1

NER_lr = 2e-5
NER_batch_size = 20
NER_num_train_epochs = 4


#Relation params
relation_label2id = {'is_a':0,'reverse_is_a':1,'is_part_of':2,'reverse_is_part_of':3,'may_treat':4,'reverse_may_treat':5,
                    'found_in':6,'reverse_found_in':7,'may_cause':8,'reverse_may_cause':9,'expressed_in':10,
                    'is_expression_of':11,'encodes':12,'encoded_by':13,'significant_drug_interaction':14,'involved_in_biological_process':15,
                    'biological_process_involves':16,'is_active_ingredient_in':17,'has_active_ingredient':18,'NULL':19
}

relation_num_labels = len(relation_label2id)
relation_id2label = {v: k for k, v in relation_label2id.items()}
relation_test_ratio = 0.1
relation_negative_ratio = 0.2
relation_common_negative_ratio = 0.3

relation_confidence_num = 3
relation_confidence_prob = 0.97


relation_lr = 2e-5
relation_batch_size = 20
relation_pred_batch_size = 200
relation_num_train_epochs = 4