from main_transfer_learning_with_finetuning.skeleton import HyperparamSearch

from train_utils.dataset import initialize_datasets_unigrams
from model.logistic_regression import build_logistic_regression_model

if __name__ == '__main__':
    hps = HyperparamSearch(
                           default_hyperparam=None,
                           default_hyperparam_features=None,
                           get_model_fn=build_logistic_regression_model,
                           get_data_generator_fn=initialize_datasets_unigrams)

    hps.main()
