from main_transfer_learning_without_finetuning.skeleton import HyperparamSearch

from train_utils.dataset import initialize_datasets_unigrams
from model.neural_network import build_neural_network_model

# The optimization config:
config = {
    "algorithm": "bayes",
    "name": "Optimize LSTM Network",
    "spec": {"maxCombo": 10, "objective": "minimize", "metric": "eRisk_UAR"},
    "parameters": {
        "chunk_size": {
            "type": "integer",
            "min": 1,
            "max": 800,
            "scalingType": "uniform",
        },
        "dense_units": {
            "type": "integer",
            "min": 1,
            "max": 800,
            "scalingType": "uniform",
        },
        "dropout_rate": {
            "type": "float",
            "min": 0.0,
            "max": 0.5,
            "scalingType": "uniform",
        },
        "batch_size": {"type": "discrete", "values": [1, 2, 3, 4, 8, 16, 32]},
        "positive_class_weight": {"type": "discrete", "values": [1, 2, 3]},
        "learning_rate": {"type": "discrete", "values": [0.1, 0.01, 0.001, 0.0005]},
    },
    "trials": 1,
}

hyperparams = {
    "positive_class_weight": 1,  #
    "chunk_size": 10,  #
    "batch_size": 32,  #
    "dense_units": 100,
    "epochs": 100,

    "threshold": 0.5,

    "reduce_lr_factor": 0.9,
    "reduce_lr_patience": 1,
    "scheduled_reduce_lr_freq": 1,
    "scheduled_reduce_lr_factor": 0.9,
    "optimizer": "adam",
    "learning_rate": 0.005,
    "early_stopping_patience": 5
}

hyperparams_features = {
    "vocabulary_path": f"../../../resources/generated/vocab_daic/unigrams_participant_3123.txt",
    "embedding_dim": 3125,
    "model": "logistic_regression",
    "embeddings_name": "unigrams"
}


def extract_from_experiment_fn(_experiment, _hyperparams, _hyperparams_features):
    _hyperparams["positive_class_weight"] = _experiment.get_parameter("positive_class_weight")
    _hyperparams["chunk_size"] = _experiment.get_parameter("chunk_size")
    _hyperparams["batch_size"] = _experiment.get_parameter("batch_size")
    _hyperparams["learning_rate"] = _experiment.get_parameter("learning_rate")
    _hyperparams["dense_units"] = _experiment.get_parameter("dense_units")
    _hyperparams["dropout_rate"] = _experiment.get_parameter("dropout_rate")


if __name__ == '__main__':
    hps = HyperparamSearch(config=config,
                           default_hyperparam=hyperparams,
                           default_hyperparam_features=hyperparams_features,
                           get_model_fn=build_neural_network_model,
                           get_data_generator_fn=initialize_datasets_unigrams,
                           extract_from_experiment_fn=extract_from_experiment_fn)
    hps.main()
