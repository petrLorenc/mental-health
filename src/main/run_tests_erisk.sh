#!/bin/bash

PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/logistic_regression/unigrams.py --dataset eRisk --hp /home/petlor/mental-health/code/resources/models/neural_network_unigrams_0_0.6889136934872938.hp.json --hpf /home/petlor/mental-health/code/resources/models/neural_network_unigrams_0_0.6889136934872938.hpf.json
PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/logistic_regression/bigrams.py --dataset eRisk --hp /home/petlor/mental-health/code/resources/models/neural_network_unigrams_0_0.6889136934872938.hp.json --hpf /home/petlor/mental-health/code/resources/models/neural_network_unigrams_0_0.6889136934872938.hpf.json

#PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/neural_network/unigrams.py --dataset eRisk
#PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/neural_network/bigrams.py --dataset eRisk

PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/bi_lstm/use4.py --dataset eRisk --hp /home/petlor/mental-health/code/resources/models/lstm_use4.vstack_0_0.7055141220236454.hp.json --hpf /home/petlor/mental-health/code/resources/models/lstm_use4.vstack_0_0.7055141220236454.hpf.json --gpu 1
PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/bi_lstm/use5.py --dataset eRisk --hp /home/petlor/mental-health/code/resources/models/lstm_use5.vstack_0_0.10295626422396087.hp.json --hpf /home/petlor/mental-health/code/resources/models/lstm_use5.vstack_0_0.10295626422396087.hpf.json --gpu 0
PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/bi_lstm/bert.py --dataset eRisk

PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/han/glove.py --dataset eRisk