from utils.logger import logger

from tensorflow.keras.utils import Sequence
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing import sequence
from resource_loading import load_NRC, load_LIWC, load_vocabulary, load_stopwords
from utils.feature_encoders import encode_emotions, encode_pronouns, encode_stopwords, encode_liwc_categories
from loader.data_loading import load_erisk_data


class EriskDataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, user_level_data, subjects_split, set_type,
                 hyperparams_features,
                 batch_size, seq_len,
                 compute_liwc=False,
                 max_posts_per_user=10,
                 pronouns=["i", "me", "my", "mine", "myself"],
                 shuffle=True,
                 keep_last_batch=True,
                 keep_first_batches=False,
                 ablate_emotions=False, ablate_liwc=False):
        # Data initialization
        self.set = set_type
        self.data = user_level_data
        self.subjects_split = subjects_split

        # Hyperparameter for data
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.pronouns = pronouns
        self.compute_liwc = compute_liwc
        self.keep_last_batch = keep_last_batch
        self.shuffle = shuffle
        self.max_posts_per_user = max_posts_per_user
        self.padding = "pre"
        self.pad_value = 0
        self.keep_first_batches = keep_first_batches  # in the rolling window case, whether it will keep

        # Initialization of utils
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

        self.stopwords_list = load_stopwords(hyperparams_features['stopwords_path'])

        self._post_indexes_per_user()
        self.on_epoch_end()

    def _post_indexes_per_user(self):
        self.indexes_per_user = {u: [] for u in range(len(self.subjects_split[self.set]))}
        self.indexes_with_user = []
        for u in range(len(self.subjects_split[self.set])):
            if self.subjects_split[self.set][u] not in self.data:
                logger.warning("User %s has no posts in %s set. Ignoring.\n" % (self.subjects_split[self.set][u], self.set))
                continue
            user_posts = self.data[self.subjects_split[self.set][u]]['texts']

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

    def __encode_text__(self, tokens):
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
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        post_indexes_per_user = {}
        # Sample post ids
        for u, post_indexes in indexes:
            user = self.subjects_split[self.set][u]
            if user in post_indexes_per_user:
                post_indexes_per_user[user].append(post_indexes)
            else:
                post_indexes_per_user[user] = [post_indexes]

        X, y = self.__data_generation_hierarchical__(post_indexes_per_user)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = self.indexes_with_user
        #         np.arange(len(self.subjects_split[self.set]))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation_hierarchical__(self, post_indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        user_tokens = []
        user_categ_data = []
        user_sparse_data = []

        labels = []
        for subject in post_indexes.keys():
            all_words = []
            all_raw_texts = []
            liwc_scores = []

            # PHQ8_Binary
            if 'label' in self.data[subject]:
                label = self.data[subject]['label']
            else:
                label = None

            for post_index_range in post_indexes[subject]:
                # Sample
                tokens = [self.data[subject]['texts'][i] for i in post_index_range]
                if 'liwc' in self.data[subject] and not self.compute_liwc:
                    liwc_selection = [self.data[subject]['liwc'][i] for i in post_index_range]
                raw_texts = [self.data[subject]['raw'][i] for i in post_index_range]

                all_words.append(tokens)
                if 'liwc' in self.data[subject] and not self.compute_liwc:
                    liwc_scores.append(liwc_selection)
                all_raw_texts.append(raw_texts)

            for i, words in enumerate(all_words):
                tokens_data = []
                categ_data = []
                sparse_data = []

                raw_text = all_raw_texts[i]
                words = all_words[i]

                for p, posting in enumerate(words):
                    raw_text_for_post = raw_text[p]
                    encoded_tokens, encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc, = self.__encode_text__(words[p])
                    if 'liwc' in self.data[subject] and not self.compute_liwc:
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
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)

        for _id in self.indexes_per_user.keys():
            if not len(self.indexes_per_user[_id]):
                logger.warning(f"User with {_id}/{self.subjects_split[self.set][_id]} does not have test data")
                continue
            user_tokens = []
            user_categ_data = []
            user_sparse_data = []

            subject = self.subjects_split[self.set][_id]
            all_words = []
            all_raw_texts = []
            liwc_scores = []

            # PHQ8_Binary
            if 'label' in self.data[subject]:
                label = self.data[subject]['label']
            else:
                label = None

            for post_index_range in self.indexes_per_user[_id]:
                # Sample
                tokens = [self.data[subject]['texts'][i] for i in post_index_range]
                if 'liwc' in self.data[subject] and not self.compute_liwc:
                    liwc_selection = [self.data[subject]['liwc'][i] for i in post_index_range]
                raw_texts = [self.data[subject]['raw'][i] for i in post_index_range]

                all_words.append(tokens)
                if 'liwc' in self.data[subject] and not self.compute_liwc:
                    liwc_scores.append(liwc_selection)
                all_raw_texts.append(raw_texts)

            for i, words in enumerate(all_words):
                tokens_data = []
                categ_data = []
                sparse_data = []

                words = all_words[i]

                for p, posting in enumerate(words):
                    encoded_tokens, encoded_emotions, encoded_pronouns, encoded_stopwords, encoded_liwc, = self.__encode_text__(words[p])
                    if 'liwc' in self.data[subject] and not self.compute_liwc:
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
