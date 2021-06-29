from utils.logger import logger

import re
import string
import random
import pickle
import numpy as np

from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import sequence
from resource_loading import load_NRC, load_LIWC, load_vocabulary, load_stopwords
from utils.feature_encoders import encode_emotions, encode_pronouns, encode_stopwords, encode_liwc_categories


class DAICDataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 hyperparams_features,
                 batch_size, seq_len,
                 compute_liwc=False,
                 max_posts_per_user=10,
                 pronouns=["i", "me", "my", "mine", "myself"],
                 shuffle=True,
                 keep_last_batch=True,
                 keep_first_batches=False,
                 ablate_emotions=False, ablate_liwc=False
                 ):
        'Initialization'
        self.seq_len = seq_len
        # Instantiate tokenizer
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.user_level_texts = {}
        self.batch_size = batch_size
        self.pronouns = pronouns
        self.compute_liwc = compute_liwc
        self.keep_last_batch = keep_last_batch
        self.shuffle = shuffle
        self.max_posts_per_user = max_posts_per_user
        self.padding = "pre"
        self.pad_value = 0
        self.keep_first_batches = keep_first_batches  # in the rolling window case, whether it will keep

        self.vocabulary = load_vocabulary(hyperparams_features['vocabulary_path'])
        self.voc_size = hyperparams_features['max_features']
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
        self.stopwords_list = load_stopwords(hyperparams_features['stopwords_path'])

    def load_daic_data(self, session, nickname=None):
        if nickname is not None:
            nickname = str(nickname)
        else:
            nickname = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))

        if nickname not in self.user_level_texts.keys():
            self.user_level_texts[nickname] = {}
            self.user_level_texts[nickname]['texts'] = []
            self.user_level_texts[nickname]['raw'] = []
            self.user_level_texts[nickname]["label"] = []

        for datapoint in session["transcripts"]:
            words = []
            raw_text = ""
            # if datapoint["speaker"] == "Participant":
            if "value" in datapoint:
                tokenized_text = self.tokenizer.tokenize(datapoint["value"])
                words.extend(tokenized_text)
                raw_text += datapoint["value"]

                self.user_level_texts[nickname]['texts'].append(words)
                self.user_level_texts[nickname]['raw'].append(raw_text)
        self.user_level_texts[nickname]["label"].append(int(session["label"]["PHQ8_Binary"]))

    def generate_indexes(self):
        self.indexes_per_user = {u: [] for u in self.user_level_texts.keys()}
        self.indexes_with_user = []
        for u, data in self.user_level_texts.items():
            user_posts = data['texts']

            # Rolling window of datapoints: chunks with overlapping posts
            nr_post_groups = len(user_posts)

            if self.keep_first_batches:
                # Generate datapoints for first posts, before a complete chunk
                for i in range(1, min(self.max_posts_per_user, nr_post_groups - 1)):
                    self.indexes_per_user[u].append(range(0, i))
                    self.indexes_with_user.append((u, range(0, i)))

            for i in range(nr_post_groups):
                # Stop at the last complete chunk
                if i + self.max_posts_per_user > len(user_posts):
                    break
                self.indexes_per_user[u].append(range(i, min(i + self.max_posts_per_user, len(user_posts))))
                self.indexes_with_user.append((u, range(i, min(i + self.max_posts_per_user, len(user_posts)))))
        self.on_epoch_end()

    def __encode_text__(self, tokens, raw_text):
        # Using 1 value for UNK token
        encoded_tokens = [self.vocabulary.get(w, 1) for w in tokens]
        encoded_emotions = encode_emotions(tokens, self.emotion_lexicon, self.emotions)
        encoded_pronouns = encode_pronouns(tokens, self.pronouns)
        encoded_stopwords = encode_stopwords(tokens, self.stopwords_list)
        if not self.compute_liwc:
            encoded_liwc = None
        else:
            encoded_liwc = encode_liwc_categories(tokens, self.liwc_categories, self.liwc_words_for_categories)

        return encoded_tokens, encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.keep_last_batch:
            return int(np.ceil(len(self.indexes) / self.batch_size))  # + 1 to not discard last batch
        return int((len(self.indexes)) / self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation_hierarchical__(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.indexes_with_user
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation_hierarchical__(self, indexes: list):
        """
        Generates data containing batch_size samples
        Args:
            indexes: ()

        Returns:

        """
        ''  # X : (n_samples, *dim, n_channels)
        user_tokens = []
        user_categ_data = []
        user_sparse_data = []

        labels = []
        for user, range_indexes in indexes:

            all_words = []
            all_raw_texts = []
            liwc_scores = []

            # PHQ8_Binary
            if 'label' in self.user_level_texts[user]:
                label = self.user_level_texts[user]['label']
            else:
                label = None

            # Sample
            texts = [self.user_level_texts[user]['texts'][i] for i in range_indexes]
            if 'liwc' in self.user_level_texts[user] and not self.compute_liwc:
                liwc_selection = [self.user_level_texts[user]['liwc'][i] for i in range_indexes]
            raw_texts = [self.user_level_texts[user]['raw'][i] for i in range_indexes]

            all_words.append(texts)
            if 'liwc' in self.user_level_texts[user] and not self.compute_liwc:
                liwc_scores.append(liwc_selection)
            all_raw_texts.append(raw_texts)

            for i, words in enumerate(all_words):
                tokens_data = []
                categ_data = []
                sparse_data = []

                raw_text = all_raw_texts[i]
                words = all_words[i]

                for p, posting in enumerate(words):
                    encoded_tokens, encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc, \
                        = self.__encode_text__(words[p], raw_text[p])
                    if 'liwc' in self.user_level_texts[user] and not self.compute_liwc:
                        liwc = liwc_scores[i][p]
                    else:
                        liwc = encoded_liwc

                    tokens_data.append(encoded_tokens)

                    categ_data.append(encoded_emotions + [encoded_pronouns] + liwc)
                    sparse_data.append(encoded_stopwords)

                # For each range
                tokens_data_padded = np.array(sequence.pad_sequences(tokens_data, maxlen=self.seq_len,
                                                                     padding=self.padding,
                                                                     truncating=self.padding))
                user_tokens.append(tokens_data_padded)

                user_categ_data.append(categ_data)
                user_sparse_data.append(sparse_data)

                labels.append(label)

        user_tokens = sequence.pad_sequences(user_tokens,
                                             maxlen=self.max_posts_per_user,
                                             value=self.pad_value)
        user_tokens = np.rollaxis(np.dstack(user_tokens), -1)
        user_categ_data = sequence.pad_sequences(user_categ_data,
                                                 maxlen=self.max_posts_per_user,
                                                 value=self.pad_value, dtype='float32')
        user_categ_data = np.rollaxis(np.dstack(user_categ_data), -1)

        user_sparse_data = sequence.pad_sequences(user_sparse_data,
                                                  maxlen=self.max_posts_per_user,
                                                  value=self.pad_value)
        user_sparse_data = np.rollaxis(np.dstack(user_sparse_data), -1)

        labels = np.array(labels, dtype=np.float32)

        return (user_tokens, user_categ_data, user_sparse_data), labels

    def yield_data_for_user(self):
        for user, index_array in self.indexes_per_user.items():
            if len(index_array) == 0:
                continue

            user_tokens = []
            user_categ_data = []
            user_sparse_data = []

            # PHQ8_Binary
            if 'label' in self.user_level_texts[user]:
                label = self.user_level_texts[user]['label']
            else:
                label = None

            for range_indexes in index_array:
                if len(index_array) == 0:
                    continue

                all_words = []
                all_raw_texts = []
                liwc_scores = []

                # Sample
                texts = [self.user_level_texts[user]['texts'][i] for i in range_indexes]
                if 'liwc' in self.user_level_texts[user] and not self.compute_liwc:
                    liwc_selection = [self.user_level_texts[user]['liwc'][i] for i in range_indexes]
                raw_texts = [self.user_level_texts[user]['raw'][i] for i in range_indexes]

                all_words.append(texts)
                if 'liwc' in self.user_level_texts[user] and not self.compute_liwc:
                    liwc_scores.append(liwc_selection)
                all_raw_texts.append(raw_texts)

                for i, words in enumerate(all_words):
                    tokens_data = []
                    categ_data = []
                    sparse_data = []

                    raw_text = all_raw_texts[i]
                    words = all_words[i]

                    for p, posting in enumerate(words):
                        encoded_tokens, encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc, \
                            = self.__encode_text__(words[p], raw_text[p])
                        if 'liwc' in self.user_level_texts[user] and not self.compute_liwc:
                            liwc = liwc_scores[i][p]
                        else:
                            liwc = encoded_liwc

                        tokens_data.append(encoded_tokens)

                        categ_data.append(encoded_emotions + [encoded_pronouns] + liwc)
                        sparse_data.append(encoded_stopwords)

                    # For each range
                    tokens_data_padded = np.array(sequence.pad_sequences(tokens_data, maxlen=self.seq_len,
                                                                         padding=self.padding,
                                                                         truncating=self.padding))
                    user_tokens.append(tokens_data_padded)

                    user_categ_data.append(categ_data)
                    user_sparse_data.append(sparse_data)

            user_tokens = sequence.pad_sequences(user_tokens,
                                                 maxlen=self.max_posts_per_user,
                                                 value=self.pad_value)
            user_tokens = np.rollaxis(np.dstack(user_tokens), -1)
            user_categ_data = sequence.pad_sequences(user_categ_data,
                                                     maxlen=self.max_posts_per_user,
                                                     value=self.pad_value, dtype='float32')
            user_categ_data = np.rollaxis(np.dstack(user_categ_data), -1)

            user_sparse_data = sequence.pad_sequences(user_sparse_data,
                                                      maxlen=self.max_posts_per_user,
                                                      value=self.pad_value)
            user_sparse_data = np.rollaxis(np.dstack(user_sparse_data), -1)

            yield label, (user_tokens, user_categ_data, user_sparse_data)