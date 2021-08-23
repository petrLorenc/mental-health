from main_transfer_learning_with_finetuning.skeleton import HyperparamSearch

from train_utils.dataset import initialize_datasets_precomputed_vector_sequence
from model.vector_precomputed import build_attention_lstm_with_vector_input_precomputed


if __name__ == '__main__':
    hps = HyperparamSearch(default_hyperparam=None,
                           default_hyperparam_features=None,
                           get_model_fn=build_attention_lstm_with_vector_input_precomputed,
                           get_data_generator_fn=initialize_datasets_precomputed_vector_sequence)
    hps.main()
