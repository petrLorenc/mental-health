import os
import pickle as plk
import numpy as np

from tensorflow.keras.preprocessing import sequence
from utils.resource_loading import load_NRC, load_dict_from_file, load_list_from_file, load_LIWC
from utils.feature_encoders import encode_emotions, encode_pronouns, encode_stopwords, encode_liwc_categories, LIWC_vectorizer
from loader.AbstractDataGenerator import AbstractDataGenerator


class DataGeneratorHierarchicalPrecomputed(AbstractDataGenerator):
    """Generates data for Keras"""

    def __init__(self, user_level_data, subjects_split, set_type, hyperparams_features, batch_size, max_seq_len,
                 precomputed_vectors_path, feature_extraction_name, embedding_dimension,
                 emotions_dim, stopwords_list_dim, liwc_categories_dim,
                 chunk_size=10, shuffle=True, keep_last_batch=True, keep_first_batches=False,
                 ablate_emotions=False, ablate_liwc=False, data_generator_id=""):

        self.precomputed_vectors_path = precomputed_vectors_path
        self.feature_extraction_name = feature_extraction_name
        self.embedding_dimension = embedding_dimension
        self.emotions_dim = emotions_dim
        self.stopwords_list_dim = stopwords_list_dim
        self.liwc_categories_dim = liwc_categories_dim

        self.pronouns = ["i", "me", "my", "mine", "myself"]

        if ablate_emotions:
            self.emotions = []
        else:
            self.emotion_lexicon = load_NRC(hyperparams_features['nrc_lexicon_path'])
            self.emotions = list(self.emotion_lexicon.keys())

        if ablate_liwc:
            self.liwc_vectorizer = LIWC_vectorizer({}, [], {})
        else:
            self.liwc_vectorizer = LIWC_vectorizer(*load_LIWC(hyperparams_features['liwc_path']))

        self.stopwords_list = load_list_from_file(hyperparams_features['stopwords_path'])

        super().__init__(user_level_data=user_level_data, subjects_split=subjects_split, set_type=set_type, batch_size=batch_size,
                         max_seq_len=max_seq_len, chunk_size=chunk_size, shuffle=shuffle,
                         keep_last_batch=keep_last_batch, keep_first_batches=keep_first_batches, data_generator_id=data_generator_id)

    def __encode_text__(self, tokens):
        encoded_emotions = encode_emotions(tokens, self.emotion_lexicon, self.emotions)
        encoded_pronouns = encode_pronouns(tokens, self.pronouns)
        encoded_stopwords = encode_stopwords(tokens, self.stopwords_list)
        encoded_liwc = encode_liwc_categories(tokens, self.liwc_vectorizer)

        return encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc

    def get_features_for_user_in_data_range(self, user, data_range):
        with open(os.path.join(self.precomputed_vectors_path, user + f".feat.{self.feature_extraction_name}.{self.embedding_dimension}.plk"), "rb") as f:
            precomputed_features_pairs = plk.load(f)
        tuple_token_features = [precomputed_features_pairs[i] for i in data_range]

        features_data = []
        categ_data = []
        sparse_data = []
        for batch in tuple_token_features:
            words_features = []
            tokens = []
            for sentence_data in batch:
                token, feature = sentence_data[0], sentence_data[1]
                tokens.append(token)
                words_features.append(feature)

            encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc, = self.__encode_text__(tokens)
            categ_data.append(encoded_emotions + [encoded_pronouns] + encoded_liwc)
            sparse_data.append(encoded_stopwords)
            features_data.append(words_features)

        return np.array(features_data).reshape((1, self.max_seq_len, self.embedding_dimension)), \
               np.array(categ_data).reshape((1, self.emotions_dim + 1 + self.liwc_categories_dim)), \
               np.array(sparse_data).reshape((1, self.stopwords_list_dim))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        features_tokens = np.zeros(shape=(len(indexes), self.chunk_size, self.max_seq_len, self.embedding_dimension))
        features_categ = np.zeros(shape=(len(indexes), self.chunk_size, self.emotions_dim + 1 + self.liwc_categories_dim))
        features_stopwords = np.zeros(shape=(len(indexes), self.chunk_size, self.stopwords_list_dim))

        labels = []

        # print(features_tokens.shape)
        # print(len(indexes))


        for idx_batch, (user, range_indexes) in enumerate(indexes):
            # PHQ8 binary
            labels.append(self.data[user]['label'] if "label" in self.data[user] else 0)

            # Get features
            with open(os.path.join(self.precomputed_vectors_path, user + f".feat.{self.feature_extraction_name}.{self.embedding_dimension}.plk"), "rb") as f:
                precomputed_features_pairs = plk.load(f)
            tuple_token_features = [precomputed_features_pairs[i] for i in range_indexes]

            # print(len(tuple_token_features))

            for idx_seq, batch in enumerate(tuple_token_features):
                tokens = []
                # print(len(batch))
                for idx_word, sentence_data in enumerate(batch):
                    if idx_word >= self.max_seq_len:
                        break
                    token, feature = sentence_data[0], sentence_data[1]
                    tokens.append(token)
                    # print(idx_batch, " ", idx_seq, " ", idx_word)
                    features_tokens[idx_batch, idx_seq, idx_word] = feature

                encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc, = self.__encode_text__(tokens)
                features_categ[idx_batch, idx_seq] = np.array(encoded_emotions + [encoded_pronouns] + encoded_liwc, dtype=np.float32)
                features_stopwords[idx_batch, idx_seq] = np.array(encoded_stopwords, dtype=np.float32)

        labels = np.array(labels, dtype=np.float32).reshape(len(indexes), 1)

        return (features_tokens, features_categ, features_stopwords), labels

    def get_data_for_specific_user(self, user):
        with open(os.path.join(self.precomputed_vectors_path, user + f".feat.{self.feature_extraction_name}.{self.embedding_dimension}.plk"), "rb") as f:
            precomputed_features_pairs = plk.load(f)
        # print(len(self.indexes_per_user[user]))

        idx_batch = 0
        for range_indexes in self.indexes_per_user[user]:
            features_tokens = np.zeros(shape=(1, self.chunk_size, self.max_seq_len, self.embedding_dimension))
            features_categ = np.zeros(shape=(1, self.chunk_size, self.emotions_dim + 1 + self.liwc_categories_dim))
            features_stopwords = np.zeros(shape=(1, self.chunk_size, self.stopwords_list_dim))

            # Get features
            tuple_token_features = [precomputed_features_pairs[i] for i in range_indexes]
            for idx_seq, batch in enumerate(tuple_token_features):
                tokens = []
                # print(len(batch))
                for idx_word, sentence_data in enumerate(batch):
                    if idx_word >= self.max_seq_len:
                        break
                    token, feature = sentence_data[0], sentence_data[1]
                    tokens.append(token)
                    # print(idx_batch, " ", idx_seq, " ", idx_word)
                    features_tokens[idx_batch, idx_seq, idx_word] = feature

                encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc, = self.__encode_text__(tokens)
                features_categ[idx_batch, idx_seq] = np.array(encoded_emotions + [encoded_pronouns] + encoded_liwc, dtype=np.float32)
                features_stopwords[idx_batch, idx_seq] = np.array(encoded_stopwords, dtype=np.float32)

            # data, label, data_identification
            yield features_tokens, features_categ, features_stopwords

