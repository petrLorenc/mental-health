import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from loader.AbstractDataGenerator import AbstractDataGenerator
from utils.feature_encoders import encode_emotions, encode_liwc_categories, LIWC_vectorizer
from utils.resource_loading import load_NRC, load_list_from_file, load_LIWC


class DataGeneratorUnigramsFeatures(AbstractDataGenerator):
    """Generates data for Keras"""

    def __init__(self, user_level_data, subjects_split, set_type, hyperparams_features, batch_size, vectorizer=None, data_generator_id=""):

        if vectorizer is None:
            vectorizer_vocabulary = load_list_from_file(hyperparams_features["vocabulary_path"])
            self.vectorizer = CountVectorizer(vocabulary=vectorizer_vocabulary, ngram_range=(1, 1))
        else:
            self.vectorizer = vectorizer

        self.tokenizer = self.vectorizer.build_tokenizer()

        self.emotion_lexicon = load_NRC(hyperparams_features['nrc_lexicon_path'])
        self.emotions = list(self.emotion_lexicon.keys())
        self.liwc_vectorizer = LIWC_vectorizer(*load_LIWC(hyperparams_features['liwc_path']))

        super().__init__(user_level_data=user_level_data, subjects_split=subjects_split, set_type=set_type, batch_size=batch_size,
                         seq_len=None, max_posts_per_user=None, shuffle=False,
                         keep_last_batch=True, keep_first_batches=True, data_generator_id=data_generator_id)

    def __encode_texts__(self, tokens):
        # Using 1 value for UNK token
        encoded_emotions = encode_emotions(tokens, self.emotion_lexicon, self.emotions)
        encoded_liwc = encode_liwc_categories(tokens, self.liwc_vectorizer)

        return encoded_emotions, encoded_liwc

    def on_data_loaded(self):
        for u in self.subjects_split[self.set]:
            if u in self.data:
                self.indexes_with_user.append((u, None))

        self.on_epoch_end()

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        features_unigrams = []
        features_emotions = []
        features_liwc = []

        labels = []
        for user, range_indexes in indexes:
            # PHQ8 binary
            labels.append(self.data[user]['label'] if "label" in self.data[user] else None)
            user_texts = " ".join([x for x in self.data[user]["raw"]])
            features_unigrams.append(self.vectorizer.transform([user_texts]).toarray().reshape(1, -1))
            tokens = self.tokenizer.tokenize(user_texts)

            encoded_emotions, encoded_liwc = self.__encode_texts__(tokens)
            features_emotions.append(encoded_emotions)
            features_liwc.append(encoded_liwc)

        labels = np.array(labels, dtype=np.float32)

        return (np.array(features_unigrams).reshape(self.batch_size, -1),
                np.array(features_emotions).reshape(self.batch_size, -1),
                np.array(features_liwc).reshape(self.batch_size, -1)), labels

    def get_data_for_specific_user(self, user):
        user_texts = " ".join([x for x in self.data[user]["raw"]])
        features_unigrams = self.vectorizer.transform([user_texts]).toarray()
        tokens = self.tokenizer.tokenize(user_texts)

        encoded_emotions, encoded_liwc = self.__encode_texts__(tokens)
        features_emotions = encoded_emotions
        features_liwc = encoded_liwc

        yield (np.array(features_unigrams).reshape(self.batch_size, -1),
               np.array(features_emotions).reshape(self.batch_size, -1),
               np.array(features_liwc).reshape(self.batch_size, -1))

    def __len__(self):
        return len(self.indexes_per_user) - 1

    def get_input_dimension(self):
        if self.vectorizer is not None:
            return len(self.vectorizer.get_feature_names())
        else:
            raise RuntimeError("Vectorizer is not set yet")
