cd ..

python main.py --dataset daic-woz --model lstm_str_dan --embeddings use-str --epochs 50 --only_test False --smaller_data False --version 1 --note daic-woz_use_tran_trainable

python main.py --dataset eRisk --model lstm_str_dan --embeddings use-str --epochs 1 --only_test False --smaller_data False --version 1 --note eRisk_use_tran_trainable