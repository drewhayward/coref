#! /bin/bash
python propara_to_jsonlines.py /data/hlr/updated_propara_data/test_propara.json test_propara_formatted.jsonlines data/spanbert_large/vocab.txt
python propara_to_jsonlines.py /data/hlr/updated_propara_data/train_propara.json train_propara_formatted.jsonlines data/spanbert_large/vocab.txt
python propara_to_jsonlines.py /data/hlr/updated_propara_data/dev_propara.json dev_propara_formatted.jsonlines data/spanbert_large/vocab.txt