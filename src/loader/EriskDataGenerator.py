from utils.logger import logger

import sys
from nltk.tokenize import RegexpTokenizer
from src.loader.DataGenerator import DataGenerator
import logging
from src.loader.data_loading import load_erisk_server_data

class EriskDataGenerator(DataGenerator):
    def __init__(self, **kwargs):
        self.data = {}
        self.subjects_split = {'test': []}
        self.tokenizer = RegexpTokenizer(r'\w+')
        super().__init__(self.data, self.subjects_split, set_type='test', **kwargs)
        
    def add_data_round(self, jldata_round):
        user_level_texts, subjects_split = load_erisk_server_data(jldata_round, self.tokenizer)
        for u in user_level_texts:
            if u not in self.data:
                self.data[u] = {k: [] for k in user_level_texts[u].keys()}
            for k in user_level_texts[u].keys():
                self.data[u][k].extend(user_level_texts[u][k])
        self.subjects_split['test'].extend(subjects_split['test'])
        self.subjects_split['test'] = list(set(self.subjects_split['test']))
        self._post_indexes_per_user()
        self.on_epoch_end()
    
    def __getitem__(self, index):
        if len(self.data) == 0:
            logger.error("Cannot generate with zero data.\n")
            return
        if len(self.data) <  self.posts_per_group:
            logger.warning("Number of input datapoints (%d) lower than minimum number of posts per chunk (%d).\n" % (len(self.data), self.posts_per_group))
        return super().__getitem__(index)