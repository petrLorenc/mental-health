cd ..

python main.py --dataset daic-woz --model lstm_distillbert --epochs 50 --only_test False --smaller_data False --version 1

python main.py --dataset eRisk --model lstm_distillbert --epochs 50 --only_test False --smaller_data False --version 1