import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from loader.AbstractDataGenerator import AbstractDataGenerator
from utils.resource_loading import load_list_from_file


class DataGeneratorUnigrams(AbstractDataGenerator):
    """Generates data for Keras"""

    def __init__(self, user_level_data, subjects_split, set_type, hyperparams_features, batch_size, chunk_size, keep_first_batches, vectorizer=None, data_generator_id=""):

        if vectorizer is None:
            vectorizer_vocabulary = load_list_from_file(hyperparams_features["vocabulary_path"])
            self.vectorizer = CountVectorizer(vocabulary=vectorizer_vocabulary, ngram_range=(1, 1))
        else:
            self.vectorizer = vectorizer

        super().__init__(user_level_data=user_level_data, subjects_split=subjects_split, set_type=set_type, batch_size=batch_size,
                         max_seq_len=None, chunk_size=chunk_size, shuffle=False,
                         keep_last_batch=True, keep_first_batches=keep_first_batches, data_generator_id=data_generator_id)

    def get_features_for_user_in_data_range(self, user, data_range):
        # data_range not used for BoW
        user_texts = " ".join([self.data[user]["raw"][x] for x in data_range])

        return self.vectorizer.transform([user_texts]).toarray().reshape((len(self.vectorizer.get_feature_names()),))

    def get_data_for_specific_user(self, user):
        o = []
        for data_range in self.indexes_per_user[user]:
            user_texts = " ".join([self.data[user]["raw"][x] for x in data_range])
            o.append(self.vectorizer.transform([user_texts]).toarray().reshape(1, -1))
        yield np.array(o).reshape((-1, len(self.vectorizer.get_feature_names())))

    def get_input_dimension(self):
        if self.vectorizer is not None:
            return len(self.vectorizer.get_feature_names())
        else:
            raise RuntimeError("Vectorizer is not set yet")
