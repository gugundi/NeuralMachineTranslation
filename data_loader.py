import itertools
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import torchtext


import re
import spacy


# load tokenizers for german and english
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


def tokenize_dummy(text):
    return text.split(' ')


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
    train.to_csv('.data/iwslt/de-en/debug.train.csv', index=False)
    val.to_csv('.data/iwslt/de-en/debug.val.csv', index=False)


def load_debug(config, SOS_token, EOS_token):
    if not os.path.exists('.data/iwslt/de-en/debug.train.csv'):
        create_debug_csv()
    source_field = torchtext.data.Field(tokenize=tokenize_de, init_token=SOS_token, eos_token=EOS_token)
    target_field = torchtext.data.Field(tokenize=tokenize_en, init_token=SOS_token, eos_token=EOS_token)
    data_fields = [('src', source_field), ('trg', target_field)]
    train, val = torchtext.data.TabularDataset.splits(
        path='.data/iwslt/de-en/',
        train='debug.train.csv',
        validation='debug.val.csv',
        format='csv',
        fields=data_fields
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
    train.to_csv('.data/iwslt/de-en/dummy_fixed_length.train.csv', index=False)
    val.to_csv('.data/iwslt/de-en/dummy_fixed_length.val.csv', index=False)


def load_dummy_fixed_length(config, SOS_token, EOS_token):
    if not os.path.exists('.data/iwslt/de-en/dummy_fixed_length.train.csv'):
        create_dummy_fixed_length_csv()
    source_field = torchtext.data.Field(tokenize=tokenize_dummy, init_token=SOS_token, eos_token=EOS_token)
    target_field = torchtext.data.Field(tokenize=tokenize_dummy, init_token=SOS_token, eos_token=EOS_token)
    data_fields = [('src', source_field), ('trg', target_field)]
    train, val = torchtext.data.TabularDataset.splits(
        path='.data/iwslt/de-en/',
        train='dummy_fixed_length.train.csv',
        validation='dummy_fixed_length.val.csv',
        format='csv',
        fields=data_fields
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


# TODO: variable length data loader


def load_iwslt(config, SOS_token, EOS_token):
    print("Started data-loader: IWSLT (de-en)")

    # set up fields for IWSLT
    DE_IWSLT = torchtext.data.Field(tokenize=tokenize_de, init_token=SOS_token, eos_token=EOS_token)
    EN_IWSLT = torchtext.data.Field(tokenize=tokenize_en, init_token=SOS_token, eos_token=EOS_token)

    print("Making splits for IWSLT")
    # make splits for data in IWSLT
    train_iwslt, val_iwslt, test_iwslt = torchtext.datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE_IWSLT, EN_IWSLT))

    print("Building vocabulary for IWSLT")
    # build the vocabulary of IWSLT
    source_vocabulary_size = config.get('source_vocabulary_size')
    target_vocabulary_size = config.get('target_vocabulary_size')
    # minus 4 for SOS, EOS, PAD and UNK
    DE_IWSLT.build_vocab(train_iwslt.src, max_size=source_vocabulary_size - 4)
    EN_IWSLT.build_vocab(train_iwslt.trg, max_size=target_vocabulary_size - 4)

    print("Making iterator splits for IWSLT")
    # make iterator for splits in IWSLT
    train_iter_iwslt, val_iter_iwslt = torchtext.data.BucketIterator.splits((train_iwslt, val_iwslt), batch_size=1, device=-1, shuffle=True)

    print("Finished loading IWSLT")

    return train_iter_iwslt, val_iter_iwslt, DE_IWSLT.vocab, EN_IWSLT.vocab


"""
# set up fields for Multi30K
DE_Multi30K = data.Field(tokenize=tokenize_x('de'))
EN_Multi30K = data.Field(tokenize=tokenize_x('en'))

# make splits for data in Multi30K
train_multi30k, val_multi30k, test_multi30k = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE_Multi30K, EN_Multi30K))

# build the vocabulary of Multi30K
DE_Multi30K.build_vocab(train_multi30k.scr, min_freq=3)
EN_Multi30K.build_vocab(train_multi30k.trg, max_size=20000)

# make iterator for splits in Multi30K
train_iter_multi30k, val_iter_multi30k = data.BucketIterator.splits(
    (train_multi30k, val_multi30k), batch_size=3, device=0)

print(DE_IWSLT.vocab.freqs.most_common(10))
print(len(DE_IWSLT.vocab))
print(EN_IWSLT.vocab.freqs.most_common(10))
print(len(EN_IWSLT.vocab))

"""
