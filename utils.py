from device import with_cpu
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


def create_iwslt():
    with open('.data/iwslt/de-en/train.de-en.de') as f:
        source = f.readlines()
        source = map(lambda sentence: sentence.replace('\n', ''), source)
        source = filter(lambda sentence: sentence != '', source)
        source = list(source)
    with open('.data/iwslt/de-en/train.de-en.en') as f:
        target = f.readlines()
        target = map(lambda sentence: sentence.replace('\n', ''), target)
        target = filter(lambda sentence: sentence != '', target)
        target = list(target)
    data = {"src": source, "trg": target}
    dataframe = pd.DataFrame(data, columns=['src', 'trg'])
    train, val = train_test_split(dataframe, test_size=0.1)
    train.to_csv('.data/iwslt/train.csv', index=False)
    val.to_csv('.data/iwslt/val.csv', index=False)


def create_multi30k():
    with open('.data/multi30k/train.en') as f:
        train_src = f.readlines()
        train_src = map(lambda sentence: sentence.replace('\n', ''), train_src)
        train_src = filter(lambda sentence: sentence != '', train_src)
        train_src = list(train_src)
    with open('.data/multi30k/train.de') as f:
        train_trg = f.readlines()
        train_trg = map(lambda sentence: sentence.replace('\n', ''), train_trg)
        train_trg = filter(lambda sentence: sentence != '', train_trg)
        train_trg = list(train_trg)
    train = {"src": train_src, "trg": train_trg}
    train = pd.DataFrame(train, columns=['src', 'trg'])

    with open('.data/multi30k/val.en') as f:
        val_src = f.readlines()
        val_src = map(lambda sentence: sentence.replace('\n', ''), val_src)
        val_src = filter(lambda sentence: sentence != '', val_src)
        val_src = list(val_src)
    with open('.data/multi30k/val.de') as f:
        val_trg = f.readlines()
        val_trg = map(lambda sentence: sentence.replace('\n', ''), val_trg)
        val_trg = filter(lambda sentence: sentence != '', val_trg)
        val_trg = list(val_trg)
    val = {"src": val_src, "trg": val_trg}
    val = pd.DataFrame(val, columns=['src', 'trg'])

    with open('.data/multi30k/test2016.en') as f:
        test_src = f.readlines()
        test_src = map(lambda sentence: sentence.replace('\n', ''), test_src)
        test_src = filter(lambda sentence: sentence != '', test_src)
        test_src = list(test_src)
    with open('.data/multi30k/test2016.de') as f:
        test_trg = f.readlines()
        test_trg = map(lambda sentence: sentence.replace('\n', ''), test_trg)
        test_trg = filter(lambda sentence: sentence != '', test_trg)
        test_trg = list(test_trg)
    test = {"src": test_src, "trg": test_trg}
    test = pd.DataFrame(test, columns=['src', 'trg'])

    train.to_csv('.data/multi30k/train.csv', index=False)
    val.to_csv('.data/multi30k/val.csv', index=False)
    test.to_csv('.data/multi30k/test.csv', index=False)


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


def load_from_csv(config, csv_dir_path, source_tokenizer, target_tokenizer, device):
    print(f'Data loader: Started ({csv_dir_path}).')

    EOS_token = config.get('EOS_token')
    PAD_token = config.get('PAD_token')
    SOS_token = config.get('SOS_token')

    source_field = torchtext.data.Field(
        tokenize=source_tokenizer,
        init_token=SOS_token,
        eos_token=EOS_token,
        pad_token=PAD_token,
        include_lengths=True
    )
    target_field = torchtext.data.Field(
        tokenize=target_tokenizer,
        init_token=SOS_token,
        eos_token=EOS_token,
        pad_token=PAD_token,
        include_lengths=True
    )
    data_fields = [('src', source_field), ('trg', target_field)]

    print('Data loader: Making splits.')
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

    print('Data loader: Building vocabulary.')
    source_field.build_vocab(train, val, max_size=source_vocabulary_size)
    target_field.build_vocab(train, val, max_size=target_vocabulary_size)

    print('Data loader: Iterator splits splits.')
    train_iter, val_iter = torchtext.data.BucketIterator.splits(
        (train, val),
        batch_size=config.get('batch_size'),
        device=device,
        shuffle=True,
        sort_key=lambda x: len(x.src)
    )

    print('Data loader: Finished.')

    return train_iter, val_iter, source_field.vocab, target_field.vocab, val


def list2words(language, sentence):
    sentence = map(lambda idx: language.itos[idx], sentence)
    return list(sentence)


def torch2words(language, sentence):
    sentence = sentence.squeeze()
    sentence = with_cpu(sentence)
    sentence = map(lambda idx: language.itos[idx], sentence)
    return list(sentence)


def filter_words(words, SOS_token, EOS_token, PAD_token):
    return filter(lambda word: word != SOS_token and word != EOS_token and word != PAD_token, words)


def words2text(words, SOS_token, EOS_token, PAD_token):
    sentence = filter_words(words, SOS_token, EOS_token, PAD_token)
    sentence = " ".join(sentence)
    return sentence


def get_text(source_words, target_words, translation_words, SOS_token, EOS_token, PAD_token):
    source = words2text(source_words, SOS_token, EOS_token, PAD_token)
    target = words2text(target_words, SOS_token, EOS_token, PAD_token)
    translation = words2text(translation_words, SOS_token, EOS_token, PAD_token)
    return f"""
    Source: \"{source}\"
    Target: \"{target}\"
    Translation: \"{translation}\"
    """
