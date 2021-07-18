from utils.logger import logger

import numpy as np

from loader.AbstractDataGenerator import AbstractDataGenerator
import os
import pickle as plk


class DataGeneratorPrecomputedVectorSequence(AbstractDataGenerator):
    """Generates data for Keras"""

    def __init__(self, user_level_data, subjects_split, set_type, batch_size, seq_len, max_posts_per_user, data_generator_id, precomputed_vectors_path, feature_extraction_name, shuffle,
                 embedding_dimension):
        super().__init__(user_level_data=user_level_data, subjects_split=subjects_split, set_type=set_type, batch_size=batch_size,
                         seq_len=seq_len, max_posts_per_user=max_posts_per_user, data_generator_id=data_generator_id, shuffle=shuffle)

        self.precomputed_vectors_path = precomputed_vectors_path
        self.feature_extraction_name = feature_extraction_name
        self.embedding_dimension = embedding_dimension

    def get_features_for_user_in_data_range(self, user, data_range):
        with open(os.path.join(self.precomputed_vectors_path, user + f".feat.{self.feature_extraction_name}.{self.embedding_dimension}.plk"), "rb") as f:
            precomputed_features = plk.load(f)
        vectors = [precomputed_features[i] for i in data_range]

        if len(vectors) == 0:
            return np.zeros(shape=(self.max_posts_per_user, self.embedding_dimension))

        vectors = np.array(vectors, dtype=np.float32)
        vectors.resize((self.max_posts_per_user, self.embedding_dimension), refcheck=False)

        return vectors

    def get_data_for_specific_user(self, user):
        for indexes in self.indexes_per_user[user]:
            with open(os.path.join(self.precomputed_vectors_path, user + f".feat.{self.feature_extraction_name}.{self.embedding_dimension}.plk"), "rb") as f:
                precomputed_features = plk.load(f)
            vectors = np.array([precomputed_features[i] for i in indexes])

            if len(vectors) == 0:
                yield np.zeros(shape=(1, self.max_posts_per_user, self.embedding_dimension))
            else:
                vectors.resize((1, self.max_posts_per_user, self.embedding_dimension), refcheck=False)
                yield np.array(vectors, dtype=np.float32)
