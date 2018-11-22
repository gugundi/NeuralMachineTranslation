from torchtext import data, datasets

import re
import spacy


# Might want to use sklearn's LabelEncoder & OneHotEncoder for one-of-K encoding
# Otherwise Keras to_categorical provides a very compact solution for integer data
# - see https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/


def load_iwslt(config, SOS_token, EOS_token):
    print("Started data-loader: IWSLT (de-en)")

    # load tokenizers for german and english
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [token.text for token in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [token.text for token in spacy_en.tokenizer(url.sub('@URL@', text))]

    # set up fields for IWSLT
    DE_IWSLT = data.Field(tokenize=tokenize_de, init_token=SOS_token, eos_token=EOS_token)
    EN_IWSLT = data.Field(tokenize=tokenize_en, init_token=SOS_token, eos_token=EOS_token)

    print("Making splits for IWSLT")
    # make splits for data in IWSLT
    train_iwslt, val_iwslt, test_iwslt = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE_IWSLT, EN_IWSLT))

    print("Building vocabulary for IWSLT")
    # build the vocabulary of IWSLT
    source_vocabulary_size = config.get('source_vocabulary_size')
    target_vocabulary_size = config.get('target_vocabulary_size')
    # minus 4 for SOS, EOS, PAD and UNK
    DE_IWSLT.build_vocab(train_iwslt.src, max_size=source_vocabulary_size - 4)
    EN_IWSLT.build_vocab(train_iwslt.trg, max_size=target_vocabulary_size - 4)

    print("Making iterator splits for IWSLT")
    # make iterator for splits in IWSLT
    # train_iter_iwslt, val_iter_iwslt = data.BucketIterator.splits((train_iwslt, val_iwslt), batch_size=config.get('batch_size'), device=-1, shuffle=True)
    train_iter_iwslt, val_iter_iwslt = data.BucketIterator.splits((train_iwslt, val_iwslt), batch_size=config.get('batch_size'))

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
