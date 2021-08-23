#!/bin/bash

PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/logistic_regression/unigrams.py --dataset alex --hp /home/petlor/mental-health/code/src/main/logistic_regression/configs/unigrams.hp.json --hpf /home/petlor/mental-health/code/src/main/logistic_regression/configs/unigrams.hpf.json --gpu 1

PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/neural_network/unigrams.py --dataset alex --hp /home/petlor/mental-health/code/src/main/neural_network/configs/unigrams.json --hpf /home/petlor/mental-health/code/src/main/neural_network/configs/unigrams.hpf.json --gpu 1


PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/bi_lstm/use4.py --dataset alex --hp /home/petlor/mental-health/code/src/main/bi_lstm/configs/use_4.hp.json --hpf /home/petlor/mental-health/code/src/main/bi_lstm/configs/use_4.hpf.json --gpu 1
PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/bi_lstm/use4_a.py --dataset alex --hp /home/petlor/mental-health/code/src/main/bi_lstm/configs/use_4.hp.json --hpf /home/petlor/mental-health/code/src/main/bi_lstm/configs/use_4.hpf.json --gpu 1

PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/bi_lstm/use5.py --dataset alex --hp /home/petlor/mental-health/code/src/main/bi_lstm/configs/use_5.hp.json --hpf /home/petlor/mental-health/code/src/main/bi_lstm/configs/use_5.hpf.json --gpu 1
PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/bi_lstm/use5_a.py --dataset alex --hp /home/petlor/mental-health/code/src/main/bi_lstm/configs/use_5.hp.json --hpf /home/petlor/mental-health/code/src/main/bi_lstm/configs/use_5.hpf.json --gpu 1

PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/bi_lstm/bert.py --dataset alex --hp /home/petlor/mental-health/code/src/main/bi_lstm/configs/bert.hp.json --hpf /home/petlor/mental-health/code/src/main/bi_lstm/configs/bert.hpf.json --gpu 1
PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/bi_lstm/bert_a.py --dataset alex --hp /home/petlor/mental-health/code/src/main/bi_lstm/configs/bert.hp.json --hpf /home/petlor/mental-health/code/src/main/bi_lstm/configs/bert.hpf.json --gpu 1

PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/han/glove.py --dataset alex --hp /home/petlor/mental-health/code/src/main/han/configs/glove.hp.json --hpf /home/petlor/mental-health/code/src/main/han/configs/glove.hpf.json --gpu 1
