import re
import Levenshtein

import pandas as pd
from pandarallel import pandarallel
import torchtext

pandarallel.initialize(nb_workers=64)


GLOVE = torchtext.vocab.GloVe(name='6B', dim=300, cache="../.vector_cache")


def text_preprocessing(text):
    try:
        text = str(text)
        text = re.sub('[,.""():，。、：；？！…“”#]', ' ', text)
        text = text.lower()  # lower
        text = text.strip()
        text = re.sub("\d+", "", text)
        text = re.sub("[\[\[\*\*##\*\*\]]", "", text)
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                      'url', text, flags=re.MULTILINE)  # replace https://... to url
        # text = re.sub('[\u00a3-\ufb04]', "", text)
        text = text.split()
    except AttributeError as e:
        return []
    return text


def abstract_title_combine(abstract_text, title):
    if abstract_text == 'NO_CONTENT':
        return title
    else:
        try:
            abstract_text = abstract_text.strip()
            title = title.strip()
            if Levenshtein.distance(abstract_text, title) >= 50:  # 差异比较大
                return title + abstract_text
            else:
                if len(abstract_text.split()) <= 1:
                    return title
                return abstract_text
        except AttributeError:
            return title


def word_to_index(words):
    index_lst = []
    assert type(words) == list
    for word in words:
        try:
            index_lst.append(GLOVE.stoi[word])
        except KeyError as e:
            pass
    return index_lst


def preprocessing(text):
    text = text_preprocessing(text)
    if text is not None:
        return word_to_index(text)
    else:
        return None


def generate_train_data(
    train_data_path="../data/train_release.csv",
    all_data_path="../data/candidate_paper_for_wsdm2020.csv",
    output_train_path="../data/train.csv",
    output_test_path="../data/test.csv",
    all_data_handed_path="../data/candidate_paper_for_wsdm2020_handed.csv",
):
    train_data = pd.read_csv(train_data_path)
    train_data.drop(index=40889, inplace=True)  # 删除不合法的行

    all_data = pd.read_csv(all_data_path, low_memory=False)
    all_data['title_abstract'] = all_data.parallel_apply(
        lambda s: abstract_title_combine(s['abstract'], s['title']), axis=1)

    all_train_merge = pd.merge(train_data, all_data, on='paper_id')
    labeled_data = all_train_merge.copy()

    all_data['title_abstract'] = all_data['title_abstract'].parallel_map(
        preprocessing)
    all_data['title_abstract'].dropna(inplace=True)
    all_data.to_csv(all_data_handed_path, index=False)

    labeled_data['title_abstract'] = labeled_data['title_abstract'].parallel_map(
        preprocessing)
    labeled_data['description_text'] = labeled_data['description_text'].parallel_map(
        preprocessing)
    labeled_data['description_text'].dropna(inplace=True)
    labeled_data['title_abstract'].dropna(inplace=True)
    labeled_data['title_abstract_len'] = labeled_data['title_abstract'].map(
        len)
    labeled_data['description_text_len'] = labeled_data['description_text'].map(
        len)
    labeled_data = labeled_data[labeled_data['description_text_len'] >= 10]
    labeled_data = labeled_data[labeled_data['title_abstract_len'] >= 10]
    labeled_data = labeled_data[[
        'title_abstract', 'description_text'
    ]]

    length = len(labeled_data)
    print("all_train_merge length：", length)
    train_data = labeled_data.iloc[:int(length*0.9)]
    test_data = labeled_data.iloc[int(length*0.9):]
    train_data.to_csv(output_train_path, index=False)
    test_data.to_csv(output_test_path, index=False)


def generate_submit_data(
    valid_data_path="../data/validation.csv",
    valid_data_handed_path="../data/validation_handed.csv",
):
    valid_data = pd.read_csv(valid_data_path)
    valid_data['description_text'] = valid_data['description_text'].parallel_map(
        preprocessing)
    valid_data.to_csv(valid_data_handed_path, index=False)


if __name__ == '__main__':
    # generate_train_data()
    generate_submit_data()
