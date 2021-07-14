python main.py --dataset daic --model neural_network --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic/bigrams_3138.txt --note daic_bigrams_3138
python main.py --dataset daic --model neural_network --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic/bigrams_16841.txt --note daic_bigrams_16841

python main.py --dataset erisk --model neural_network --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic/bigrams_3138.txt --note daic_bigrams_3138
python main.py --dataset erisk --model neural_network --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic/bigrams_16841.txt --note daic_bigrams_16841


python main.py --dataset erisk --model neural_network --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_erisk/bigrams_20000.txt --note erisk_bigrams_20000

python main.py --dataset daic --model neural_network --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_erisk/bigrams_20000.txt --note erisk_bigrams_20000


python main.py --dataset daic --model neural_network --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/bigrams_daic_bigrams_3138_erisk_bigrams_20000_resulting_20721.txt --note daic_bigrams_daic_bigrams_3138_erisk_bigrams_20000_resulting_20721
python main.py --dataset daic --model neural_network --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/bigrams_daic_bigrams_16841_erisk_bigrams_20000_resulting_29088.txt --note daic_bigrams_daic_bigrams_16841_erisk_bigrams_20000_resulting_29088


python main.py --dataset erisk --model neural_network --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/bigrams_daic_bigrams_3138_erisk_bigrams_20000_resulting_20721.txt --note erisk_bigrams_daic_bigrams_3138_erisk_bigrams_20000_resulting_20721
python main.py --dataset erisk --model neural_network --embeddings bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/bigrams_daic_bigrams_16841_erisk_bigrams_20000_resulting_29088.txt --note erisk_bigrams_daic_bigrams_16841_erisk_bigrams_20000_resulting_29088
