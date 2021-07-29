"""
Each file with vectors will have name of subject will be saved as pickle
"""
import os
import numpy as np
import pickle as plk
from nltk.tokenize import RegexpTokenizer

import tensorflow as tf
import tensorflow_hub as hub

from loader.data_loading import load_erisk_data, load_daic_data
from utils.resource_loading import load_NRC, load_dict_from_file, load_list_from_file, load_LIWC
from utils.feature_encoders import encode_emotions, encode_pronouns, encode_stopwords, encode_liwc_categories, LIWC_vectorizer

import argparse

from spacy.lang.en import English

nlp = English()
# Create a Tokenizer with the default settings for English
# including punctuation rules and exceptions
tokenizer = nlp.tokenizer


def __encode_text__(tokens, emotion_lexicon, emotions, pronouns, stopwords_list, liwc_vectorizer):
    # Using 1 value for UNK token
    encoded_emotions = encode_emotions(tokens, emotion_lexicon, emotions)
    encoded_pronouns = encode_pronouns(tokens, pronouns)
    encoded_stopwords = encode_stopwords(tokens, stopwords_list)
    encoded_liwc = encode_liwc_categories(tokens, liwc_vectorizer)

    return encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--dataset', type=str)

    liwc_path = "../resources/liwc.dic"
    nrc_path = "../resources/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
    stopword_path = "../resources/stopwords.txt"
    args = parser.parse_args()

    rewrite = True
    dataset = args.dataset

    dir_to_save = f"../data/{dataset}/precomputed_features/"

    print(f"Checking directory to save features {dir_to_save}")
    if not os.path.exists(dir_to_save):
        os.mkdir(dir_to_save)

    stopwords_list = load_list_from_file(stopword_path)
    pronouns = ["i", "me", "my", "mine", "myself"]
    liwc_vectorizer = LIWC_vectorizer(*load_LIWC(liwc_path))
    emotion_lexicon = load_NRC(nrc_path)
    emotions = list(emotion_lexicon.keys())

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

        encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc = [], [], [], []
        for raw_text in raw_texts:
            tokens = [x.text for x in tokenizer(raw_text)]
            emo, pron, stopw, liwc = __encode_text__(tokens=tokens,
                                                     emotion_lexicon=emotion_lexicon,
                                                     stopwords_list=stopwords_list,
                                                     emotions=emotions,
                                                     pronouns=pronouns,
                                                     liwc_vectorizer=liwc_vectorizer)
            encoded_emotions.append(emo)
            encoded_pronouns.append(pron)
            encoded_stopwords.append(stopw)
            encoded_liwc.append(liwc)

        output_mapping = {
            "emotions": encoded_emotions,
            "pronouns": encoded_pronouns,
            "stopwords": encoded_stopwords,
            "liwc": encoded_liwc
        }

        for aggregation_choice, vectors in output_mapping.items():
            with open(os.path.join(dir_to_save, k + f".feat.{aggregation_choice}.plk"), "wb") as f:
                plk.dump(vectors, f)
            print(f"{k} preprocessed - {aggregation_choice}")
