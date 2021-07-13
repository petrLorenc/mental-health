python train.py --dataset daic --model lstm_str --embeddings use-str --epochs 50 --only_test False --smaller_data False --version 1 --note daic_use_trainable
 > ../resources/logs/use_nontrainable.daic.log 2>&1

python train.py --dataset erisk --model lstm_str --embeddings use-str --epochs 50 --only_test False --smaller_data False --version 1 --note erisk_use_trainable
 > ../resources/logs/use_nontrainable.erisk.log 2>&1