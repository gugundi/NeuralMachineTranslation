import os
import spacy
import torchtext
from utils import (
    create_debug_csv,
    create_multi30k,
    create_iwslt,
    create_dummy_fixed_length_csv,
    create_dummy_variable_length_csv,
    get_or_create_dir,
    load_from_csv
)


# load tokenizers for german and english
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]


def tokenize_dummy(text):
    return text.split(' ')


def load_debug(config, device):
    csv_dir_path = get_or_create_dir('.data', 'debug')
    if not os.path.exists(f'{csv_dir_path}/train.csv'):
        create_debug_csv()
    return load_from_csv(config, csv_dir_path, tokenize_de, tokenize_en, device)


def load_dummy_fixed_length(config, device):
    csv_dir_path = get_or_create_dir('.data', 'dummy_fixed_length')
    if not os.path.exists(f'{csv_dir_path}/train.csv'):
        create_dummy_fixed_length_csv()
    return load_from_csv(config, csv_dir_path, tokenize_dummy, tokenize_dummy, device)


def load_dummy_variable_length(config, device):
    csv_dir_path = get_or_create_dir('.data', 'dummy_variable_length')
    if not os.path.exists(f'{csv_dir_path}/train.csv'):
        create_dummy_variable_length_csv()
    return load_from_csv(config, csv_dir_path, tokenize_dummy, tokenize_dummy, device)


def load_iwslt(config, device):
    csv_dir_path = get_or_create_dir('.data', 'iwslt')
    if not os.path.exists(f'{csv_dir_path}/train.csv'):
        if not os.path.exists(f'{csv_dir_path}/de-en'):
            source_field = torchtext.data.Field(tokenize=tokenize_de)
            target_field = torchtext.data.Field(tokenize=tokenize_en)
            torchtext.datasets.IWSLT.splits(exts=('.de', '.en'), fields=(source_field, target_field))
        create_iwslt()
    return load_from_csv(config, csv_dir_path, tokenize_de, tokenize_en, device)


def load_multi30k(config, device):
    csv_dir_path = get_or_create_dir('.data', 'multi30k')
    if not os.path.exists(f'{csv_dir_path}/train.csv'):
        source_field = torchtext.data.Field(tokenize=tokenize_de)
        target_field = torchtext.data.Field(tokenize=tokenize_en)
        torchtext.datasets.Multi30k.splits(exts=('.de', '.en'), fields=(source_field, target_field))
        create_multi30k()
    return load_from_csv(config, csv_dir_path, tokenize_de, tokenize_en, device)
