"""
Each file with vectors will have name of subject will be saved as pickle
"""
import os
import numpy as np
import pickle as plk
from nltk.tokenize import RegexpTokenizer

import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer

from loader.data_loading import load_erisk_data, load_daic_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[1], enable=True)

def get_aggregation_fn(choice="words"):
    if choice == "words":
        return np.array
    else:
        raise Exception(f"Unknown aggregation fn choice: {choice}")

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--code', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--dimension', type=str)
    args = parser.parse_args()

    rewrite = True
    processing_batch = 10
    dataset = args.dataset

    code_name = args.code
    feature_extraction = args.name
    embedding_dim = args.dimension

    dir_to_save = f"../data/{dataset}/precomputed_features/"

    print(f"Checking directory to save features {dir_to_save}")
    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)

    print("Loading feature extractor...")
    tokenizer = AutoTokenizer.from_pretrained(feature_extraction)
    vectorizer = TFAutoModel.from_pretrained(feature_extraction)

    feature_extraction = feature_extraction.replace('/', '-')

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
        # if os.path.isfile(os.path.join(dir_to_save, k + f".feat.{code_name}.{aggregation_choice}.{embedding_dim}.plk")) and rewrite is False:
        #     print(f"Skipping user {k}")
        #     continue
        raw_texts = user_level_data[k]["raw"]

        num_batches = len(raw_texts) // processing_batch
        preprocessed_vectors = []

        for idx in range(num_batches + 1):
            if len(raw_texts[idx * processing_batch: (idx + 1) * processing_batch]) > 0:
                try:
                    input_to_model = tokenizer(raw_texts[idx * processing_batch: (idx + 1) * processing_batch], return_tensors="tf", padding=True, truncation=True)
                    output_from_model = vectorizer(input_to_model).last_hidden_state.numpy()
                    print(output_from_model.shape)
                    for idx in range(len(output_from_model)):
                        preprocessed_vectors.append(list(zip(filter(lambda x: x != "[PAD]", tokenizer.convert_ids_to_tokens(input_to_model["input_ids"].numpy()[idx])), np.squeeze(output_from_model[idx]))))

                except Exception as e:
                    print(f"{k} has errors with {len(raw_texts)} - {e}")
                    for separate_example in raw_texts[idx * processing_batch: (idx + 1) * processing_batch]:
                        input_to_model = tokenizer([separate_example], return_tensors="tf", padding=True, truncation=True)
                        try:
                            output_from_model = vectorizer(input_to_model).last_hidden_state.numpy()
                        except Exception:
                            output_from_model = np.zeros(shape=(1, int(args.dimension)))
                        print(output_from_model.shape)

                        for idx in range(len(output_from_model)):
                            preprocessed_vectors.append(list(zip(filter(lambda x: x != "[PAD]", tokenizer.convert_ids_to_tokens(input_to_model["input_ids"].numpy()[idx])),np.squeeze(output_from_model[idx]))))

        for aggregation_choice in ["words"]:
            aggregation_fn = get_aggregation_fn(aggregation_choice)
            preprocessed_vectors_temp = aggregation_fn(preprocessed_vectors)
            print(preprocessed_vectors_temp.shape)
            with open(os.path.join(dir_to_save, k + f".feat.{code_name}.{aggregation_choice}.{embedding_dim}.plk"), "wb") as f:
                plk.dump(preprocessed_vectors_temp, f)
            print(f"{k} preprocessed - {aggregation_choice}")
