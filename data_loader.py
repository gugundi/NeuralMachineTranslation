import os
import spacy
import torchtext
from utils import create_debug_csv, create_dummy_fixed_length_csv, get_or_create_dir, load_from_csv


# load tokenizers for german and english
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


def tokenize_dummy(text):
    return text.split(' ')


def load_debug(config, SOS_token, EOS_token):
    csv_dir_path = get_or_create_dir('.data', 'debug')
    if not os.path.exists(f'{csv_dir_path}/train.csv'):
        create_debug_csv()
    return load_from_csv(config, SOS_token, EOS_token, csv_dir_path, tokenize_de, tokenize_en)


def load_dummy_fixed_length(config, SOS_token, EOS_token):
    csv_dir_path = get_or_create_dir('.data', 'dummy_fixed_length')
    if not os.path.exists(f'{csv_dir_path}/train.csv'):
        create_dummy_fixed_length_csv()
    return load_from_csv(config, SOS_token, EOS_token, csv_dir_path, tokenize_dummy, tokenize_dummy)


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
