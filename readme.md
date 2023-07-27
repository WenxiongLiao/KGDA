#  Coarse-to-fine Knowledge Graph Domain Adaptation based on Distantly-supervised Iterative Training BIBM2023 paper

## software environment

```
pip install -r requirements.txt
```

## download data
Download domain text corpus：https://pan.baidu.com/s/1rV4g8Hog7j1A4BMh9IkEpg  password:abkf
Download coarse-domain KG and pre-trained language models (PLMs)
Download coarse-domain KG (BIOS 2022v1(beta)): https://bios.idea.edu.cn/Download
Download emilyalsentzer-Bio_ClinicalBERT: https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT
Download allenai-biomed_roberta_base: https://huggingface.co/allenai/biomed_roberta_base
Download bert-base-uncased: https://huggingface.co/bert-base-uncased

Move the  domain text corpus in folder ./data
Move the  coarse-domain KG (BIOS 2022v1(beta)) in folder ./data
Move the  PLMs in folder ./save_models


## Train and held-out test
```
python KGDA-main.py
```

## Output
```
In floder "./data/AC_dict/MODEL_NAME/"

New entities: concept_semantic_confidence_post_AC.pkl
Triples with new relations: new_concept_relation_confidence_post_dict.pkl
Triples with new entities: new_relation_with_new_concept_post_dict.pkl
Overlapping entities: domain_concept_dict.pkl
Overlapping triples: domain_concept_relation_dict.pkl
```