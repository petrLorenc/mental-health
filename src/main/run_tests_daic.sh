#!/bin/bash

PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/logistic_regression/unigrams.py --dataset daic-woz
PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/logistic_regression/bigrams.py --dataset daic-woz

PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/neural_network/unigrams.py --dataset daic-woz
PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/neural_network/bigrams.py --dataset daic-woz

PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/bi_lstm/use4.py --dataset daic-woz
PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/bi_lstm/use5.py --dataset daic-woz
PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/bi_lstm/bert.py --dataset daic-woz

PYTHONPATH=/home/petlor/mental-health/code/src/ python /home/petlor/mental-health/code/src/main/han/glove.py --dataset daic-woz