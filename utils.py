import itertools
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torchtext


def get_or_create_dir(base_path, dir_name):
    out_directory = os.path.join(base_path, dir_name)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    return out_directory


def create_debug_csv():
    n_lines = 1000
    with open('.data/iwslt/de-en/train.de-en.de') as f:
        source = itertools.islice(f, n_lines)
        source = map(lambda sentence: sentence.replace('\n', ''), source)
        source = list(source)
    with open('.data/iwslt/de-en/train.de-en.en') as f:
        target = itertools.islice(f, n_lines)
        target = map(lambda sentence: sentence.replace('\n', ''), target)
        target = list(target)
    data = {"src": source, "trg": target}
    dataframe = pd.DataFrame(data, columns=['src', 'trg'])
    train, val = train_test_split(dataframe, test_size=0.1)
    train.to_csv('.data/debug/train.csv', index=False)
    val.to_csv('.data/debug/val.csv', index=False)


def create_dummy_fixed_length_csv():
    n_observations = 10000
    source_length = 10
    target_length = 5
    condition = 5
    max_int = 10
    src = []
    trg = []
    for i in range(n_observations):
        positions = range(source_length)
        source = [random.randint(condition, max_int) for _ in range(source_length)]
        for j in range(target_length):
            position = random.choice(positions)
            source[position] = random.randint(1, condition - 1)
            positions = list(set(positions) - set([position]))
        target = filter(lambda x: x < condition, source)
        source = map(str, source)
        target = map(str, target)
        src.append(" ".join(source))
        trg.append(" ".join(target))
    data = {"src": src, "trg": trg}
    dataframe = pd.DataFrame(data, columns=['src', 'trg'])
    train, val = train_test_split(dataframe, test_size=0.1)
    train.to_csv('.data/dummy_fixed_length/train.csv', index=False)
    val.to_csv('.data/dummy_fixed_length/val.csv', index=False)


def create_dummy_variable_length_csv():
    n_observations = 10000
    min_source_length = 3
    max_source_length = 15
    condition = 5
    max_int = 10
    src = []
    trg = []
    for i in range(n_observations):
        source_length = random.randint(min_source_length, max_source_length)
        source = [random.randint(1, max_int) for _ in range(source_length)]
        target = filter(lambda x: x < condition, source)
        source = map(str, source)
        target = map(str, target)
        src.append(" ".join(source))
        trg.append(" ".join(target))
    data = {"src": src, "trg": trg}
    dataframe = pd.DataFrame(data, columns=['src', 'trg'])
    train, val = train_test_split(dataframe, test_size=0.1)
    train.to_csv('.data/dummy_variable_length/train.csv', index=False)
    val.to_csv('.data/dummy_variable_length/val.csv', index=False)


def load_from_csv(config, SOS_token, EOS_token, csv_dir_path, source_tokenizer, target_tokenizer):
    source_field = torchtext.data.Field(tokenize=source_tokenizer, init_token=SOS_token, eos_token=EOS_token)
    target_field = torchtext.data.Field(tokenize=target_tokenizer, init_token=SOS_token, eos_token=EOS_token)
    data_fields = [('src', source_field), ('trg', target_field)]
    train, val = torchtext.data.TabularDataset.splits(
        path=csv_dir_path,
        train='train.csv',
        validation='val.csv',
        format='csv',
        fields=data_fields,
        skip_header=True
    )
    source_vocabulary_size = config.get('source_vocabulary_size')
    target_vocabulary_size = config.get('target_vocabulary_size')
    source_field.build_vocab(train, val, max_size=source_vocabulary_size - 4)
    target_field.build_vocab(train, val, max_size=target_vocabulary_size - 4)
    train_iter, val_iter = torchtext.data.BucketIterator.splits(
        (train, val),
        batch_size=1,
        device=-1,
        shuffle=True,
        sort_key=lambda x: len(x.src)
    )
    return train_iter, val_iter, source_field.vocab, target_field.vocab
