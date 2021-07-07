from utils.logger import logger

import json
# cleaning the text
import re
import contractions


def load_erisk_data(writings_df, valid_prop=0.2,
                    min_post_len=1, labelcol='label'):
    training_subjects = list(set(writings_df[writings_df['subset'] == 'train'].subject))
    test_subjects = list(set(writings_df[writings_df['subset'] == 'test'].subject))

    training_subjects = sorted(training_subjects)  # ensuring reproducibility
    valid_subjects_size = int(len(training_subjects) * valid_prop)
    valid_subjects = training_subjects[:valid_subjects_size]
    training_subjects = training_subjects[valid_subjects_size:]

    subjects_split = {'train': training_subjects,
                      'valid': valid_subjects,
                      'test': test_subjects}

    user_level_texts = {}
    for row in writings_df.sort_values(by='date').itertuples():
        words = []
        raw_text = ""
        if hasattr(row, 'tokenized_title'):
            if row.tokenized_title:
                words.extend(row.tokenized_title)
                raw_text += row.title
        if hasattr(row, 'tokenized_text'):
            if row.tokenized_text:
                words.extend(row.tokenized_text)
                raw_text += row.text
        if not words or len(words) < min_post_len:
            continue
        if labelcol == 'label':
            label = row.label
        if row.subject not in user_level_texts.keys():
            user_level_texts[row.subject] = {}
            user_level_texts[row.subject]['texts'] = [words]
            user_level_texts[row.subject]['label'] = label
            user_level_texts[row.subject]['raw'] = [raw_text]
        else:
            user_level_texts[row.subject]['texts'].append(words)
            user_level_texts[row.subject]['raw'].append(raw_text)

    return user_level_texts, subjects_split


def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    text = text.replace("_", "")
    text = text.replace("'s", "")
    return re.sub(pat, '', text)


def load_daic_data(path_train,
                   path_valid,
                   path_test,
                   tokenizer,
                   include_only=["Ellie", "Participant"],
                   limit_size=False):
    with open(path_train, "r") as f:
        data_train = json.load(f)
        data_train = data_train[:int((0.2 if limit_size else 1.0) * len(data_train))]
    with open(path_valid, "r") as f:
        data_valid = json.load(f)
        data_valid = data_valid[:int((0.2 if limit_size else 1.0) * len(data_valid))]
    with open(path_test, "r") as f:
        data_test = json.load(f)
        data_test = data_test[:int((0.2 if limit_size else 1.0) * len(data_test))]

    training_subjects = [x["label"]["Participant_ID"] for x in data_train]
    valid_subjects = [x["label"]["Participant_ID"] for x in data_valid]
    test_subjects = [x["label"]["Participant_ID"] for x in data_test]

    # ensuring reproducibility (with same version of sorting)
    training_subjects = sorted(training_subjects)
    valid_subjects = sorted(valid_subjects)
    test_subjects = sorted(test_subjects)

    subjects_split = {'train': training_subjects,
                      'valid': valid_subjects,
                      'test': test_subjects}

    user_level_texts = {}
    for row in data_train + data_valid + data_test:
        for utterance in row["transcripts"]:
            if not utterance["speaker"] in include_only:
                continue
            raw_text = utterance["value"]
            raw_text = remove_special_characters(raw_text)
            raw_text = contractions.fix(raw_text)

            label = int(row["label"]["PHQ8_Binary"])
            subject = row["label"]["Participant_ID"]
            if subject not in user_level_texts.keys():
                user_level_texts[subject] = {}
                user_level_texts[subject]['texts'] = tokenizer.tokenize(raw_text)
                user_level_texts[subject]['label'] = label
                user_level_texts[subject]['raw'] = [raw_text]
            else:
                user_level_texts[subject]['texts'].append(tokenizer.tokenize(raw_text))
                user_level_texts[subject]['raw'].append(raw_text)

    return user_level_texts, subjects_split


def load_erisk_server_data(dataround_json, tokenizer, verbose=0):
    if verbose:
        logger.debug("Loading data...\n")

    subjects_split = {'test': []}
    user_level_texts = {}

    for datapoint in dataround_json:
        words = []
        raw_text = ""
        if "title" in datapoint:
            tokenized_title = tokenizer.tokenize(datapoint["title"])
            words.extend(tokenized_title)
            raw_text += datapoint["title"]
        if "content" in datapoint:
            tokenized_text = tokenizer.tokenize(datapoint["content"])
            words.extend(tokenized_text)
            raw_text += datapoint["content"]

        if datapoint["nick"] not in user_level_texts.keys():
            user_level_texts[datapoint["nick"]] = {}
            user_level_texts[datapoint["nick"]]['texts'] = [words]
            user_level_texts[datapoint["nick"]]['raw'] = [raw_text]
            subjects_split['test'].append(datapoint['nick'])
        else:
            user_level_texts[datapoint["nick"]]['texts'].append(words)
            user_level_texts[datapoint["nick"]]['raw'].append(raw_text)

    return user_level_texts, subjects_split
