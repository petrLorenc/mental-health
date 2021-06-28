from utils.logger import logger

from abc import abstractmethod

import numpy as np
from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.utils import Sequence


class AbstractDataGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self,
                 vectorizer=None,
                 shuffle=True):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.user_level_texts = {}
        self.shuffle = shuffle
        self.vectorizer = vectorizer
        self.frozen_users = []

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.frozen_users = list(self.user_level_texts.keys())
        if self.shuffle:
            np.random.shuffle(self.frozen_users)

    @abstractmethod
    def yield_data_for_specific_user(self, user):
        """
        Generates data containing batch_size samples
        Args:
            indexes: ()

        Returns:

        """
        ''  # X : (n_samples, *dim, n_channels)
        label = self.user_level_texts[user]['label']
        user_texts = " ".join([x for x in self.user_level_texts[user]["raw"]])

        label = np.array(label, dtype=np.float32)

        return self.vectorizer.transform([user_texts]).toarray(), label
