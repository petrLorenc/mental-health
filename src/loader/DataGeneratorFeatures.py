from utils.logger import logger

import re
import string
import random
import pickle
import numpy as np

from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import sequence
from resource_loading import load_NRC, load_LIWC, load_vocabulary, load_list_from_file
from utils.feature_encoders import encode_emotions, encode_pronouns, encode_stopwords, encode_liwc_categories
from loader.AbstractDataGenerator import AbstractDataGenerator


class DataGeneratorHierarchical(AbstractDataGenerator):
    """Generates data for Keras"""

    def __init__(self, user_level_data, subjects_split, set_type, hyperparams_features, batch_size, seq_len,
                 max_posts_per_user=10, shuffle=True, keep_last_batch=True, keep_first_batches=False,
                 ablate_emotions=False, ablate_liwc=False):

        self.pronouns = ["i", "me", "my", "mine", "myself"]

        self.vocabulary = load_vocabulary(hyperparams_features['vocabulary_path'])
        self.voc_size = len(self.vocabulary)

        if ablate_emotions:
            self.emotions = []
        else:
            self.emotion_lexicon = load_NRC(hyperparams_features['nrc_lexicon_path'])
            self.emotions = list(self.emotion_lexicon.keys())

        self.liwc_words_for_categories = pickle.load(open(hyperparams_features["liwc_words_cached"], "rb"))
        if ablate_liwc:
            self.liwc_categories = []
        else:
            self.liwc_categories = set(self.liwc_words_for_categories.keys())

        self.stopwords_list = load_list_from_file(hyperparams_features['stopwords_path'])

        super().__init__(user_level_data, subjects_split, set_type, hyperparams_features, batch_size, seq_len, max_posts_per_user, shuffle,
                         keep_last_batch, keep_first_batches)

    def __encode_text__(self, tokens):
        # Using 1 value for UNK token
        encoded_tokens = [self.vocabulary.get(w, 1) for w in tokens]
        encoded_emotions = encode_emotions(tokens, self.emotion_lexicon, self.emotions)
        encoded_pronouns = encode_pronouns(tokens, self.pronouns)
        encoded_stopwords = encode_stopwords(tokens, self.stopwords_list)
        encoded_liwc = encode_liwc_categories(tokens, self.liwc_categories, self.liwc_words_for_categories)

        return encoded_tokens, encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.keep_last_batch:
            return int(np.ceil(len(self.indexes) / self.batch_size))  # + 1 to not discard last batch
        return int((len(self.indexes)) / self.batch_size)

    def get_features_for_user_in_data_range(self, user, data_range):
        sequence_tokens = [self.data[user]['texts'][i] for i in data_range]

        tokens_data = []
        categ_data = []
        sparse_data = []
        for sentence_tokens in sequence_tokens:
            encoded_tokens, encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc, = self.__encode_text__(sentence_tokens)
            tokens_data.append(encoded_tokens)

            categ_data.append(encoded_emotions + [encoded_pronouns] + encoded_liwc)
            sparse_data.append(encoded_stopwords)

        return tokens_data, categ_data, sparse_data

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        features_tokens = []
        features_categ = []
        features_stopwords = []

        labels = []
        for user, range_indexes in indexes:
            # PHQ8 binary
            labels.append(self.data[user]['label'] if "label" in self.data[user] else None)
            # Get features
            f_tokens, f_categ, f_stopwords = self.get_features_for_user_in_data_range(user, range_indexes)
            tokens_data_padded = np.array(sequence.pad_sequences(f_tokens, maxlen=self.seq_len,
                                                                 padding=self.padding,
                                                                 truncating=self.padding))
            features_tokens.append(tokens_data_padded)
            features_categ.append(f_categ)
            features_stopwords.append(f_stopwords)

        user_tokens = sequence.pad_sequences(features_tokens,
                                             maxlen=self.max_posts_per_user,
                                             value=self.pad_value)
        user_tokens = np.rollaxis(np.dstack(user_tokens), -1)
        user_categ_data = sequence.pad_sequences(features_categ,
                                                 maxlen=self.max_posts_per_user,
                                                 value=self.pad_value, dtype='float32')
        user_categ_data = np.rollaxis(np.dstack(user_categ_data), -1)

        user_sparse_data = sequence.pad_sequences(features_stopwords,
                                                  maxlen=self.max_posts_per_user,
                                                  value=self.pad_value)
        user_sparse_data = np.rollaxis(np.dstack(user_sparse_data), -1)

        labels = np.array(labels, dtype=np.float32)

        return (user_tokens, user_categ_data, user_sparse_data), labels

    def get_data_for_specific_user(self, user):
        label = self.data[user]['label']
        raw_texts = self.data[user]["raw"]

        features_tokens = []
        features_categ = []
        features_stopwords = []

        for range_indexes in self.indexes_per_user[user]:
            # Get features
            f_tokens, f_categ, f_stopwords = self.get_features_for_user_in_data_range(user, range_indexes)
            tokens_data_padded = np.array(sequence.pad_sequences(f_tokens, maxlen=self.seq_len,
                                                                 padding=self.padding,
                                                                 truncating=self.padding))
            features_tokens.append(tokens_data_padded)
            features_categ.append(f_categ)
            features_stopwords.append(f_stopwords)

        user_tokens = sequence.pad_sequences(features_tokens,
                                             maxlen=self.max_posts_per_user,
                                             value=self.pad_value)
        user_tokens = np.rollaxis(np.dstack(user_tokens), -1)
        user_categ_data = sequence.pad_sequences(features_categ,
                                                 maxlen=self.max_posts_per_user,
                                                 value=self.pad_value, dtype='float32')
        user_categ_data = np.rollaxis(np.dstack(user_categ_data), -1)

        user_sparse_data = sequence.pad_sequences(features_stopwords,
                                                  maxlen=self.max_posts_per_user,
                                                  value=self.pad_value)
        user_sparse_data = np.rollaxis(np.dstack(user_sparse_data), -1)

        # data, label, data_identification
        return (user_tokens, user_categ_data, user_sparse_data), label, raw_texts

