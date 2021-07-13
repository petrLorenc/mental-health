python train.py --dataset daic --model log_regression --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_5924_daic.txt --note daic_5924
 > ../resources/logs/bigrams.vocab_5924_daic.log 2>&1
python train.py --dataset daic --model log_regression --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_20000_erisk.txt --note daic_20000
 > ../resources/logs/bigrams.vocab_20000_daic.log 2>&1
python train.py --dataset daic --model log_regression --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_20997_erisk_daic.txt --note daic_20997
 > ../resources/logs/bigrams.vocab_20997_daic.log 2>&1

python train.py --dataset erisk --model log_regressiom --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_5924_daic.txt --note erisk_5924
 > ../resources/logs/bigrams.vocab_5924_erisk.log 2>&1
python train.py --dataset erisk --model log_regression --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_20000_erisk.txt --note erisk_20000
 > ../resources/logs/bigrams.vocab_20000_erisk.log 2>&1
python train.py --dataset erisk --model log_regression --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_20997_erisk_daic.txt --note erisk_20997
 > ../resources/logs/bigrams.vocab_20997_erisk.log 2>&1