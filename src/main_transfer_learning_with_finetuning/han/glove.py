from main_transfer_learning_with_finetuning.skeleton import HyperparamSearch

from train_utils.dataset import initialize_datasets_hierarchical
from model.hierarchical_model import build_hierarchical_model

embeddings = "glove"

if __name__ == '__main__':
    hps = HyperparamSearch(default_hyperparam=None,
                           default_hyperparam_features=None,
                           get_model_fn=build_hierarchical_model,
                           get_data_generator_fn=initialize_datasets_hierarchical)
    hps.main()
