cd ..

python main.py --dataset daic-woz --model neural_network_bigrams --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic-woz/bigrams_3138.txt --note daic-woz_bigrams_3138
python main.py --dataset daic-woz --model neural_network_bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz/bigrams_16841.txt --note daic-woz_bigrams_16841

python main.py --dataset eRisk --model neural_network_bigrams --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic-woz/bigrams_3138.txt --note daic-woz_bigrams_3138
python main.py --dataset eRisk --model neural_network_bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz/bigrams_16841.txt --note daic-woz_bigrams_16841


python main.py --dataset eRisk --model neural_network_bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_eRisk/bigrams_20000.txt --note eRisk_bigrams_20000

python main.py --dataset daic-woz --model neural_network_bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_eRisk/bigrams_20000.txt --note eRisk_bigrams_20000


python main.py --dataset daic-woz --model neural_network_bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/bigrams_daic-woz_bigrams_3138_eRisk_bigrams_20000_resulting_20721.txt --note daic-woz_bigrams_daic-woz_bigrams_3138_eRisk_bigrams_20000_resulting_20721
python main.py --dataset daic-woz --model neural_network_bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/bigrams_daic-woz_bigrams_16841_eRisk_bigrams_20000_resulting_29088.txt --note daic-woz_bigrams_daic-woz_bigrams_16841_eRisk_bigrams_20000_resulting_29088


python main.py --dataset eRisk --model neural_network_bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/bigrams_daic-woz_bigrams_3138_eRisk_bigrams_20000_resulting_20721.txt --note eRisk_bigrams_daic-woz_bigrams_3138_eRisk_bigrams_20000_resulting_20721
python main.py --dataset eRisk --model neural_network_bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic-woz_eRisk/bigrams_daic-woz_bigrams_16841_eRisk_bigrams_20000_resulting_29088.txt --note eRisk_bigrams_daic-woz_bigrams_16841_eRisk_bigrams_20000_resulting_29088
