"""
Each file with vectors will have name of subject will be saved as pickle
"""
import os
import numpy as np
import pickle as plk
from nltk.tokenize import RegexpTokenizer

import torch
from transformers import AutoModel, AutoTokenizer

from loader.data_loading import load_erisk_data, load_daic_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_aggregation_fn(choice="vstack"):
    if choice == "vstack":
        return np.vstack
    elif choice == "average":
        return lambda x: np.average(np.vstack(x), axis=0)
    elif choice == "maximum":
        return lambda x: np.max(np.vstack(x), axis=0)
    elif choice == "minimum":
        return lambda x: np.min(np.vstack(x), axis=0)
    else:
        raise Exception(f"Unknown aggregation fn choice: {choice}")


if __name__ == '__main__':
    rewrite = True
    processing_batch = 25
    dataset = "daic-woz"
    # dataset = "eRisk"

    as_whole = True
    aggregation_choice = "average" # does not apply if as_whole is True

    code_name = "use4"
    feature_extraction = "google/bigbird-roberta-base"

    aggregation_fn = get_aggregation_fn(aggregation_choice)

    embedding_dim = 512
    dir_to_save = f"../data/{dataset}/precomputed_features/"

    print(f"Checking directory to save features {dir_to_save}")
    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)

    print("Loading feature extractor...")
    if "bigbird" in feature_extraction:
        from transformers.models import BigBirdTokenizer, BigBirdModel

        tokenizer = BigBirdTokenizer.from_pretrained(feature_extraction)
        vectorizer = BigBirdModel.from_pretrained('google/bigbird-roberta-base')

    else:
        tokenizer = AutoTokenizer.from_pretrained(feature_extraction)
        vectorizer = AutoModel.from_pretrained(feature_extraction)

    print("Loading datasets...")
    if dataset == "eRisk":
        writings_df = plk.load(open('../data/eRisk/writings_df_depression_liwc', 'rb'))
        user_level_data, subjects_split = load_erisk_data(writings_df)

    elif dataset == "daic-woz":
        user_level_data, subjects_split = load_daic_data(path_train="../data/daic-woz/train_data.json",
                                                         path_valid="../data/daic-woz/dev_data.json",
                                                         path_test="../data/daic-woz/test_data.json",
                                                         include_only=["Participant"],
                                                         # include_only=["Ellie", "Participant"],
                                                         limit_size=False,
                                                         tokenizer=RegexpTokenizer(r'\w+'))
    else:
        raise Exception(f"Unknown dataset: {dataset}")

    for k in user_level_data.keys():
        if os.path.isfile(os.path.join(dir_to_save, k + f".feat.{code_name}.{aggregation_choice}.{embedding_dim}.plk")) and rewrite is False:
            print(f"Skipping user {k}")
            continue
        raw_texts = user_level_data[k]["raw"]

        num_batches = len(raw_texts) // processing_batch
        preprocessed_vectors = []

        if as_whole:
            preprocessed_vectors.append(model(raw_texts))
        else:
            try:
                for idx in range(num_batches + 1):
                    if len(raw_texts[idx * processing_batch: (idx + 1) * processing_batch]) > 0:
                        preprocessed_vectors.append(vectorizer(
                            tokenizer(raw_texts[idx * processing_batch: (idx + 1) * processing_batch], return_tensors="tf", padding=True,
                                      truncation=True)).last_hidden_state.numpy()[:, 0])
                preprocessed_vectors = aggregation_fn(preprocessed_vectors)
            except Exception as e:
                print(f"{k} has errors with {len(raw_texts)} - {e}")

        with open(os.path.join(dir_to_save, k + f".feat.{code_name}.{aggregation_choice}.{embedding_dim}.plk"), "wb") as f:
            plk.dump(preprocessed_vectors, f)
        print(f"{k} preprocessed")
