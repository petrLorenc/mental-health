from nltk.corpus import stopwords


def encode_emotions(tokens, emotion_lexicon, emotions, relative=True):
    encoded_emotions = [0 for _ in emotions]
    text_len = len(tokens)
    for i, emotion in enumerate(emotions):
        emotion_words = [t for t in tokens if t in emotion_lexicon[emotion]]
        if relative and len(tokens):
            encoded_emotions[i] = len(emotion_words) / len(tokens)
        else:
            encoded_emotions[i] = len(emotion_words)
        if relative and text_len:
            encoded_emotions[i] = encoded_emotions[i] / text_len

    return encoded_emotions


def encode_pronouns(tokens, pronouns={"i", "me", "my", "mine", "myself"}, relative=True):
    if not tokens:
        return 0
    text_len = len(tokens)
    nr_pronouns = len([t for t in tokens if t in pronouns])
    if relative and text_len:
        return nr_pronouns / text_len
    else:
        return nr_pronouns


def encode_stopwords(tokens, stopwords_list=None, relative=True):
    if not stopwords_list:
        stopwords_list = stopwords.words("english")
    encoded_stopwords = [0 for s in stopwords_list]
    text_len = len(tokens)
    if not tokens:
        return encoded_stopwords
    for i, stopword in enumerate(stopwords_list):
        if stopword in tokens:
            encoded_stopwords[i] += 1
        if relative and text_len:
            encoded_stopwords[i] = encoded_stopwords[i] / text_len
    return encoded_stopwords


def encode_liwc_categories(tokens, liwc_categories, liwc_words_for_categories, relative=True):
    categories_cnt = [0 for _ in liwc_categories]
    if not tokens:
        return categories_cnt
    text_len = len(tokens)
    for i, category in enumerate(liwc_categories):
        for t in tokens:
            if t in liwc_words_for_categories[category]:
                categories_cnt[i] += 1
        if relative and text_len:
            categories_cnt[i] = categories_cnt[i] / text_len
    return categories_cnt
