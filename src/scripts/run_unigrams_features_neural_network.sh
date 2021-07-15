cd ..

python main.py --dataset daic --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic/unigrams_3138.txt --note daic_unigrams_3138
python main.py --dataset daic --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic/unigrams_5924.txt --note daic_unigrams_5924

python main.py --dataset erisk --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic/unigrams_3138.txt --note daic_unigrams_3138
python main.py --dataset erisk --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic/unigrams_5924.txt --note daic_unigrams_5924


python main.py --dataset erisk --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_erisk/unigrams_20000.txt --note erisk_unigrams_20000
python main.py --dataset daic --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_erisk/unigrams_20000.txt --note erisk_unigrams_20000

python main.py --dataset erisk --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_erisk/unigrams_20000.original.txt --note erisk_unigrams_original_20000
python main.py --dataset daic --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_erisk/unigrams_20000.original.txt --note erisk_unigrams_original_20000


python main.py --dataset daic --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/unigrams_daic_5924_erisk_20000_original_resulting_20997.txt --note daic_unigrams_daic_5924_erisk_20000_original_resulting_20997
python main.py --dataset daic --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/unigrams_daic_unigrams_3138_erisk_unigrams_20000_original_resulting_20144.txt --note daic_unigrams_daic_unigrams_3138_erisk_unigrams_20000_original_resulting_20144
python main.py --dataset daic --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/unigrams_daic_unigrams_3138_erisk_unigrams_20000_resulting_20163.txt --note daic_unigrams_daic_unigrams_3138_erisk_unigrams_20000_resulting_20163
python main.py --dataset daic --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/unigrams_daic_unigrams_5924_erisk_unigrams_20000_original_resulting_20999.txt --note daic_unigrams_daic_unigrams_5924_erisk_unigrams_20000_original_resulting_20999
python main.py --dataset daic --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/unigrams_daic_unigrams_5924_erisk_unigrams_20000_resulting_21049.txt --note daic_unigrams_daic_unigrams_5924_erisk_unigrams_20000_resulting_21049


python main.py --dataset erisk --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/unigrams_daic_5924_erisk_20000_original_resulting_20997.txt --note erisk_unigrams_daic_5924_erisk_20000_original_resulting_20997
python main.py --dataset erisk --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/unigrams_daic_unigrams_3138_erisk_unigrams_20000_original_resulting_20144.txt --note erisk_unigrams_daic_unigrams_3138_erisk_unigrams_20000_original_resulting_20144
python main.py --dataset erisk --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/unigrams_daic_unigrams_3138_erisk_unigrams_20000_resulting_20163.txt --note erisk_unigrams_daic_unigrams_3138_erisk_unigrams_20000_resulting_20163
python main.py --dataset erisk --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/unigrams_daic_unigrams_5924_erisk_unigrams_20000_original_resulting_20999.txt --note erisk_unigrams_daic_unigrams_5924_erisk_unigrams_20000_original_resulting_20999
python main.py --dataset erisk --model neural_network_features --embeddings unigrams-features --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic_erisk/unigrams_daic_unigrams_5924_erisk_unigrams_20000_resulting_21049.txt --note erisk_unigrams_daic_unigrams_5924_erisk_unigrams_20000_resulting_21049
