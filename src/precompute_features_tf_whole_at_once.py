"""
Each file with vectors will have name of subject will be saved as pickle
"""
import os
import numpy as np
import random
import pickle as plk
from nltk.tokenize import RegexpTokenizer

import tensorflow as tf
from transformers import XLNetTokenizer, TFXLNetModel

from loader.data_loading import load_erisk_data, load_daic_data

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
    tf.config.experimental.set_memory_growth(device=gpus[1], enable=True)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--code', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('--dimension', type=str)
    args = parser.parse_args()

    rewrite = True
    dataset = args.dataset

    code_name = args.code
    feature_extraction = args.name
    embedding_dim = args.dimension

    dir_to_save = f"../data/{dataset}/precomputed_features/"

    print(f"Checking directory to save features {dir_to_save}")
    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)

    print("Loading feature extractor...")
    tokenizer = XLNetTokenizer.from_pretrained(feature_extraction)
    vectorizer = TFXLNetModel.from_pretrained(feature_extraction)

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
        if os.path.isfile(os.path.join(dir_to_save, k + f".feat.{code_name}.{embedding_dim}.plk")) and rewrite is False:
            print(f"Skipping user {k}")
            continue
        raw_texts = user_level_data[k]["raw"]
        whole_chunk = "".join(raw_texts)
        tokens = whole_chunk.split(" ")
        print(len(tokens))
        if len(tokens) > 2048:
            sample_size = 2048
            sorted_sample = [tokens[i] for i in sorted(random.sample(range(len(tokens)), sample_size))]
            whole_chunk = " ".join(sorted_sample)

        # (batch_size, num_predict, hidden_size)
        embedding = vectorizer(tokenizer(whole_chunk, return_tensors="tf")).last_hidden_state.numpy()[0, -1, :]  # last token is CLS
        print(embedding.shape)

        with open(os.path.join(dir_to_save, k + f".feat.{code_name}.{embedding_dim}.plk"), "wb") as f:
            plk.dump(embedding, f)
        print(f"{k} preprocessed")
