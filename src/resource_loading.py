from utils.liwc_readDict import readDict
import pickle
import numpy as np


def load_NRC(nrc_path):
    word_emotions = {}
    emotion_words = {}
    with open(nrc_path) as in_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            word, emotion, label = line.split()
            if word not in word_emotions:
                word_emotions[word] = set()
            if emotion not in emotion_words:
                emotion_words[emotion] = set()
            label = int(label)
            if label:
                word_emotions[word].add(emotion)
                emotion_words[emotion].add(word)
    return emotion_words


def load_LIWC(path):
    liwc_dict = {}
    for (w, c) in readDict(path):
        if c not in liwc_dict:
            liwc_dict[c] = []
        liwc_dict[c].append(w)
    return liwc_dict


# PAD and UNK included (index 0, 1)
def load_vocabulary(path):
    vocabulary_dict = {}
    with open(path, "r") as f:
        for i, w in enumerate(f):
            vocabulary_dict[w] = i
    return vocabulary_dict


def load_embeddings(embeddings_path, embedding_dim, vocabulary):
    # random matrix with mean value = 0
    embedding_matrix = np.random.random((len(vocabulary), embedding_dim)) - 0.5
    not_found_words_cnt = 0
    with open(embeddings_path, encoding='utf8') as f:
        for i, line in enumerate(f):
            values = line.split()
            word = values[0]
            # -embedding_dim because of tokens with spaces (like ". . .")
            coefs = np.asarray(values[-embedding_dim:], dtype='float32')
            word_i = vocabulary.get(word)
            if word_i is not None:
                embedding_matrix[word_i] = coefs
            else:
                not_found_words_cnt += 1

    print(f'Total {len(embedding_matrix)} word vectors.')
    print(f'Words not found in embedding space {not_found_words_cnt}')

    return embedding_matrix


def load_stopwords(path):
    stopwords_list = []
    with open(path) as f:
        for line in f:
            stopwords_list.append(line)
    return stopwords_list
