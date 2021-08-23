from main.skeleton import HyperparamSearch

from train_utils.dataset import initialize_datasets_precomputed_vector_sequence
from model.vector_precomputed import build_attention_lstm_with_vector_input_precomputed

embeddings = "use4.vstack"

# The optimization config:
config = {
    "algorithm": "bayes",
    "name": "Optimize LSTM Network",
    "spec": {"maxCombo": 10, "objective": "minimize", "metric": "average_CV_UAR"},
    "parameters": {
        "lstm_units": {"type": "discrete", "values": [32, 44, 56, 68, 88, 106, 128, 186, 200]},
        "chunk_size": {"type": "discrete", "values": [2, 4, 6, 8, 10, 16, 22, 32, 64, 88, 128]},
        "dropout_rate": {"type": "discrete", "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]},
        "batch_size": {"type": "discrete", "values": [2, 4, 6,  8, 10, 16, 22, 32]},
        "positive_class_weight": {"type": "discrete", "values": [1, 2, 3]},
        "learning_rate": {"type": "discrete", "values": [0.1, 0.01, 0.001, 0.0005]}
    },
    "trials": 2,
}

hyperparams = {
    "positive_class_weight": 1,  #
    "chunk_size": 10,  #
    "batch_size": 32,  #
    "epochs": 100,

    "threshold": 0.5,

    "reduce_lr_factor": 0.9,
    "reduce_lr_patience": 1,
    "scheduled_reduce_lr_freq": 1,
    "scheduled_reduce_lr_factor": 0.9,
    "optimizer": "adam",
    "learning_rate": 0.005,
    "early_stopping_patience": 5,

    "dropout_rate": 0.1,  #
    "lstm_units": 256  #
}

hyperparams_features = {
    "model": "lstm",
    "embeddings_name": embeddings,
    "embedding_dim": 512
}

def extract_from_experiment_fn(_experiment, _hyperparams, _hyperparams_features):
    _hyperparams_features["precomputed_vectors_path"] = f"/home/petlor/mental-health/code/data/{hyperparams['dataset']}/precomputed_features/"

    _hyperparams["positive_class_weight"] = _experiment.get_parameter("positive_class_weight")
    _hyperparams["chunk_size"] = _experiment.get_parameter("chunk_size")
    _hyperparams["dropout_rate"] = _experiment.get_parameter("dropout_rate")
    _hyperparams["lstm_units"] = _experiment.get_parameter("lstm_units")
    _hyperparams["batch_size"] = _experiment.get_parameter("batch_size")
    _hyperparams["learning_rate"] = _experiment.get_parameter("learning_rate")


if __name__ == '__main__':
    hps = HyperparamSearch(config=config,
                           default_hyperparam=hyperparams,
                           default_hyperparam_features=hyperparams_features,
                           get_model_fn=build_attention_lstm_with_vector_input_precomputed,
                           get_data_generator_fn=initialize_datasets_precomputed_vector_sequence,
                           extract_from_experiment_fn=extract_from_experiment_fn)
    hps.main()
