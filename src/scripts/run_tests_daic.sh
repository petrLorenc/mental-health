cd ..
#
## logistic regression variants
#python main.py --dataset daic-woz --model log_regression_unigrams --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic/unigrams_participant_3123.txt --note daic-woz-unigrams-3123
#python main.py --dataset daic-woz --model log_regression_unigrams_features --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic/unigrams_participant_3123.txt --note daic-woz-unigrams-3123
#python main.py --dataset daic-woz --model log_regression_bigrams --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic/bigrams_participant_3123.txt --note daic-woz-bigrams-3123
#python main.py --dataset daic-woz --model log_regression_bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic/bigrams_participant_16385.txt --note daic-woz-bigrams-16385
#
## neural networks variants
#python main.py --dataset daic-woz --model neural_network_unigrams --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic/unigrams_participant_3123.txt --note daic-woz-unigrams-3123
#python main.py --dataset daic-woz --model neural_network_unigrams_features --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic/unigrams_participant_3123.txt --note daic-woz-unigrams-3123
#python main.py --dataset daic-woz --model neural_network_bigrams --epochs 50 --only_test False --smaller_data False --version 1  --vocabulary ../generated/vocab_daic/bigrams_participant_3123.txt --note daic-woz-bigrams-3123
#python main.py --dataset daic-woz --model neural_network_bigrams --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic/bigrams_participant_16385.txt --note daic-woz-bigrams-16385
#
## HAN
#python main.py --dataset daic-woz --model hierarchical --epochs 50 --only_test False --smaller_data False --version 1 --vocabulary ../generated/vocab_daic/unigrams_participant_3123.txt --note daic-woz-unigrams-3123
#
## precomputed vectors
## USE4
#python precompute_features_tfh.py --dataset daic-woz --code use4 --name "../resources/embeddings/use-4" --dimension 512
#python main.py --dataset daic-woz --model precomputed_embeddings_sequence_lstm --embeddings use4.vstack --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1
#
#python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_neural_network --embeddings use4.average --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1
#python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_logistic_regression --embeddings use4.average --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1
#
#python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_neural_network --embeddings use4.maximum --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1
#python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_logistic_regression --embeddings use4.maximum --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1
#
#python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_neural_network --embeddings use4.minimum --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1
#python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_logistic_regression --embeddings use4.minimum --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1
#
## USE5
#python precompute_features_tfh.py --dataset daic-woz --code use5 --name "../resources/embeddings/use-5" --dimension 512

python main.py --dataset daic-woz --model precomputed_embeddings_sequence_lstm --embeddings use5.vstack --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1

python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_neural_network --embeddings use5.average --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1
python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_logistic_regression --embeddings use5.average --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1

python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_neural_network --embeddings use5.maximum --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1
python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_logistic_regression --embeddings use5.maximum --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1

python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_neural_network --embeddings use5.minimum --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1
python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_logistic_regression --embeddings use5.minimum --embeddings_dim 512 --epochs 50 --only_test False --smaller_data False --version 1

## XLNet
#python precompute_features_tf_whole_at_once.py --dataset daic-woz --code xlnet --name "xlnet-base-cased" --dimension 768
#python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_neural_network --embeddings xlnet --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1
#python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_logistic_regression --embeddings xlnet --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1
#
## Bertweet (<s> token at beginning and </s> token at the end)
#python precompute_features_tf.py --dataset daic-woz --code bertweet --name "vinai/bertweet-base" --dimension 768 --aggregation vstack --cls_position 0
#
#python main.py --dataset daic-woz --model precomputed_embeddings_sequence_lstm --embeddings bertweet.vstack --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1
#python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_neural_network --embeddings bertweet.average --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1
#python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_logistic_regression --embeddings bertweet.average --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1
#
#python precompute_features_tf.py --dataset daic-woz --code bertweet --name "vinai/bertweet-base" --dimension 768 --aggregation vstack --cls_position -1
#python main.py --dataset daic-woz --model precomputed_embeddings_sequence_lstm --embeddings bertweet.vstack --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1
#python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_logistic_regression --embeddings bertweet.average --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1
#python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_neural_network --embeddings bertweet.average --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1


#
## BERT
python precompute_features_tf.py --dataset daic-woz --code bert --name "bert-base-cased" --dimension 768
python main.py --dataset daic-woz --model precomputed_embeddings_sequence_lstm --embeddings bert.vstack --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1

python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_neural_network --embeddings bert.average --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1
python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_logistic_regression --embeddings bert.average --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1

python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_neural_network --embeddings bert.maximum --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1
python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_logistic_regression --embeddings bert.maximum --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1

python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_neural_network --embeddings bert.minimum --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1
python main.py --dataset daic-woz --model precomputed_embeddings_aggregated_logistic_regression --embeddings bert.minimum --embeddings_dim 768 --epochs 50 --only_test False --smaller_data False --version 1
