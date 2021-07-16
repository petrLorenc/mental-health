cd ..

python main.py --dataset daic-woz --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic-woz/unigrams_3138.txt --note daic-woz_unigrams_3138
python main.py --dataset daic-woz --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz/unigrams_5924.txt --note daic-woz_unigrams_5924

python main.py --dataset eRisk --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic-woz/unigrams_3138.txt --note daic-woz_unigrams_3138
python main.py --dataset eRisk --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz/unigrams_5924.txt --note daic-woz_unigrams_5924


python main.py --dataset eRisk --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_eRisk/unigrams_20000.txt --note eRisk_unigrams_20000
python main.py --dataset daic-woz --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_eRisk/unigrams_20000.txt --note eRisk_unigrams_20000

python main.py --dataset eRisk --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_eRisk/unigrams_20000.original.txt --note eRisk_unigrams_original_20000
python main.py --dataset daic-woz --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_eRisk/unigrams_20000.original.txt --note eRisk_unigrams_original_20000


python main.py --dataset daic-woz --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/unigrams_daic-woz_5924_eRisk_20000_original_resulting_20997.txt --note daic-woz_unigrams_daic-woz_5924_eRisk_20000_original_resulting_20997
python main.py --dataset daic-woz --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/unigrams_daic-woz_unigrams_3138_eRisk_unigrams_20000_original_resulting_20144.txt --note daic-woz_unigrams_daic-woz_unigrams_3138_eRisk_unigrams_20000_original_resulting_20144
python main.py --dataset daic-woz --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/unigrams_daic-woz_unigrams_3138_eRisk_unigrams_20000_resulting_20163.txt --note daic-woz_unigrams_daic-woz_unigrams_3138_eRisk_unigrams_20000_resulting_20163
python main.py --dataset daic-woz --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/unigrams_daic-woz_unigrams_5924_eRisk_unigrams_20000_original_resulting_20999.txt --note daic-woz_unigrams_daic-woz_unigrams_5924_eRisk_unigrams_20000_original_resulting_20999
python main.py --dataset daic-woz --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/unigrams_daic-woz_unigrams_5924_eRisk_unigrams_20000_resulting_21049.txt --note daic-woz_unigrams_daic-woz_unigrams_5924_eRisk_unigrams_20000_resulting_21049


python main.py --dataset eRisk --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/unigrams_daic-woz_5924_eRisk_20000_original_resulting_20997.txt --note eRisk_unigrams_daic-woz_5924_eRisk_20000_original_resulting_20997
python main.py --dataset eRisk --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/unigrams_daic-woz_unigrams_3138_eRisk_unigrams_20000_original_resulting_20144.txt --note eRisk_unigrams_daic-woz_unigrams_3138_eRisk_unigrams_20000_original_resulting_20144
python main.py --dataset eRisk --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/unigrams_daic-woz_unigrams_3138_eRisk_unigrams_20000_resulting_20163.txt --note eRisk_unigrams_daic-woz_unigrams_3138_eRisk_unigrams_20000_resulting_20163
python main.py --dataset eRisk --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/unigrams_daic-woz_unigrams_5924_eRisk_unigrams_20000_original_resulting_20999.txt --note eRisk_unigrams_daic-woz_unigrams_5924_eRisk_unigrams_20000_original_resulting_20999
python main.py --dataset eRisk --model neural_network --embeddings unigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/unigrams_daic-woz_unigrams_5924_eRisk_unigrams_20000_resulting_21049.txt --note eRisk_unigrams_daic-woz_unigrams_5924_eRisk_unigrams_20000_resulting_21049
