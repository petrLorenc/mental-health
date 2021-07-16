cd ..

python main.py --dataset daic-woz --model lstm_vector_dan --embeddings use-vector --epochs 50 --only_test False --smaller_data False --version 1 --note daic-woz_use_dan_nontrainable

python main.py --dataset eRisk --model lstm_vector_dan --embeddings use-vector --epochs 50 --only_test False --smaller_data False --version 1 --note eRisk_use_dan_nontrainable