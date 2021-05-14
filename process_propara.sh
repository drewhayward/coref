#! /bin/bash
python propara_to_jsonlines.py $1/test_propara.json test_propara_formatted.jsonlines data/spanbert_large/vocab.txt
python propara_to_jsonlines.py $1/train_propara.json train_propara_formatted.jsonlines data/spanbert_large/vocab.txt
python propara_to_jsonlines.py $1/dev_propara.json dev_propara_formatted.jsonlines data/spanbert_large/vocab.txt