from utils.logger import logger

import numpy as np

from loader.AbstractDataGenerator import AbstractDataGenerator
import os
import pickle as plk


class DataGeneratorPrecomputedVectorAggregated(AbstractDataGenerator):
    """Generates data for Keras"""

    def __init__(self, user_level_data, subjects_split, set_type, batch_size, seq_len, max_posts_per_user, data_generator_id, precomputed_vectors_path, feature_extraction_name, shuffle,
                 embedding_dimension):
        super().__init__(user_level_data=user_level_data, subjects_split=subjects_split, set_type=set_type, batch_size=batch_size,
                         seq_len=seq_len, max_posts_per_user=max_posts_per_user, data_generator_id=data_generator_id, shuffle=shuffle)
        self.precomputed_vectors_path = precomputed_vectors_path
        self.feature_extraction_name = feature_extraction_name
        self.embedding_dimension = embedding_dimension

    def on_data_loaded(self):
        for u in self.subjects_split[self.set]:
            if u in self.data:
                self.indexes_with_user.append((u, None))

        self.on_epoch_end()

    def get_features_for_user_in_data_range(self, user, data_range):
        with open(os.path.join(self.precomputed_vectors_path, user + f".feat.{self.feature_extraction_name}.{self.embedding_dimension}.plk"), "rb") as f:
            precomputed_features = plk.load(f)

        if len(precomputed_features) == 0:
            return np.zeros(shape=(self.embedding_dimension,))

        return np.array(precomputed_features, dtype=np.float32).reshape((self.embedding_dimension,))

    def get_data_for_specific_user(self, user):
        with open(os.path.join(self.precomputed_vectors_path, user + f".feat.{self.feature_extraction_name}.{self.embedding_dimension}.plk"), "rb") as f:
            precomputed_features = plk.load(f)

        if len(precomputed_features) == 0:
            yield np.zeros(shape=(1, self.embedding_dimension))
        else:
            yield np.array(precomputed_features, dtype=np.float32).reshape((1, self.embedding_dimension))
