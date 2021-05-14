#! /bin/bash
export data_dir=data
GPU=0 python predict.py spanbert_large train_propara_formatted.jsonlines train_propara_coref.jsonlines
GPU=0 python predict.py spanbert_large test_propara_formatted.jsonlines test_propara_coref.jsonlines
GPU=0 python predict.py spanbert_large dev_propara_formatted.jsonlines dev_propara_coref.jsonlines