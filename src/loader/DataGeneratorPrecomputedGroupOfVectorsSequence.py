from utils.logger import logger

import numpy as np

from tensorflow.keras.preprocessing import sequence

from loader.AbstractDataGenerator import AbstractDataGenerator
import os
import pickle as plk


class DataGeneratorPrecomputedGroupOfVectorsSequence(AbstractDataGenerator):
    """Generates data for Keras"""

    def __init__(self, user_level_data, subjects_split, set_type, batch_size, seq_len, max_posts_per_user, data_generator_id, precomputed_vectors_path,
                 embedding_name, shuffle,
                 embedding_dimension,
                 other_features):
        super().__init__(user_level_data=user_level_data, subjects_split=subjects_split, set_type=set_type, batch_size=batch_size,
                         seq_len=seq_len, max_posts_per_user=max_posts_per_user, data_generator_id=data_generator_id, shuffle=shuffle)

        self.precomputed_vectors_path = precomputed_vectors_path

        self.embedding_name = embedding_name
        self.embedding_dimension = embedding_dimension
        self.other_features = other_features
        self.additional_dimension = 0
        user = subjects_split["train"][0]
        for feat in self.other_features:
            with open(os.path.join(self.precomputed_vectors_path, user + f".feat.{feat}.plk"), "rb") as f:
                precomputed_features = plk.load(f)
                self.additional_dimension += len(precomputed_features[0]) if type(precomputed_features[0]) is list else 1

    def get_features_for_user_in_data_range(self, user, data_range):
        with open(os.path.join(self.precomputed_vectors_path, user + f".feat.{self.embedding_name}.{self.embedding_dimension}.plk"), "rb") as f:
            precomputed_features = plk.load(f)
        vectors = [precomputed_features[i] for i in data_range]

        if len(vectors) == 0:
            return np.zeros(shape=(self.max_posts_per_user, self.embedding_dimension))

        vectors = np.array(vectors, dtype=np.float32)
        vectors.resize((self.max_posts_per_user, self.embedding_dimension), refcheck=False)

        return vectors

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        features_embeddings = []
        features_additional = []

        labels = []
        for user, range_indexes in indexes:
            # PHQ8 binary
            labels.append(self.data[user]['label'] if "label" in self.data[user] else None)
            features_embeddings.append(self.get_features_for_user_in_data_range(user, range_indexes))

            temp_features_for_chunk = []
            for i in range_indexes:
                temp_features = []
                for feat in self.other_features:
                    with open(os.path.join(self.precomputed_vectors_path, user + f".feat.{feat}.plk"), "rb") as f:
                        precomputed_features = plk.load(f)
                    temp_features.extend(precomputed_features[i] if type(precomputed_features[i]) is list else [precomputed_features[i]])
                temp_features_for_chunk.append(temp_features)
            features_additional.append(temp_features_for_chunk)

        labels = np.array(labels, dtype=np.float32)
        features_embeddings = np.array(features_embeddings, dtype=np.float32)
        features_additional = np.array(features_additional, dtype=np.float32)

        return (features_embeddings, features_additional), labels

    def get_data_for_specific_user(self, user):
        for indexes in self.indexes_per_user[user]:
            with open(os.path.join(self.precomputed_vectors_path, user + f".feat.{self.embedding_name}.{self.embedding_dimension}.plk"), "rb") as f:
                precomputed_features = plk.load(f)
            vectors = np.array([precomputed_features[i] for i in indexes])

            features_for_chunk = []
            for i in indexes:
                temp_features = []
                for feat in self.other_features:
                    with open(os.path.join(self.precomputed_vectors_path, user + f".feat.{feat}.plk"), "rb") as f:
                        precomputed_features = plk.load(f)
                    temp_features.extend(precomputed_features[i] if type(precomputed_features[i]) is list else [precomputed_features[i]])
                features_for_chunk.append(temp_features)

            if len(vectors) == 0:
                yield np.zeros(shape=(1, self.max_posts_per_user, self.embedding_dimension))
            else:
                vectors.resize((1, self.max_posts_per_user, self.embedding_dimension), refcheck=False)
                yield np.array(vectors, dtype=np.float32), np.expand_dims(np.array(features_for_chunk, dtype=np.float32), axis=0)
