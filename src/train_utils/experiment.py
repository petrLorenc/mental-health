from comet_ml import Experiment, Optimizer

def get_network_type(hyperparams):
    if 'lstm' in hyperparams['ignore_layer']:
        network_type = 'cnn'
    else:
        network_type = 'lstm'
    if 'user_encoded' in hyperparams['ignore_layer']:
        if 'bert_layer' not in hyperparams['ignore_layer']:
            network_type = 'bert'
        else:
            network_type = 'extfeatures'
    if hyperparams['hierarchical']:
        hierarch_type = 'hierarchical'
    else:
        hierarch_type = 'seq'
    return network_type, hierarch_type


def initialize_experiment(hyperparams, nrc_lexicon_path, emotions, pretrained_embeddings_path,
                          dataset_type, transfer_type, hyperparams_features):
    # experiment = Experiment(api_key="eoBdVyznAhfg3bK9pZ58ZSXfv",
    #                         project_name="mental", workspace="ananana", disabled=False)
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="6XP0ix9zkGMuM24VbrnVRHSbf",
        project_name="general",
        workspace="petr-lorenc",
        disabled=False
    )

    experiment.log_parameters(hyperparams_features)

    experiment.log_parameter('emotion_lexicon', nrc_lexicon_path)
    experiment.log_parameter('emotions', emotions)
    experiment.log_parameter('embeddings_path', pretrained_embeddings_path)
    experiment.log_parameter('dataset_type', dataset_type)
    experiment.log_parameter('transfer_type', transfer_type)
    experiment.add_tag(dataset_type)
    experiment.log_parameters(hyperparams)
    network_type, hierarch_type = get_network_type(hyperparams)
    experiment.add_tag(network_type)
    experiment.add_tag(hierarch_type)

    return experiment