from torchtext import data,datasets,vocab
import spacy
import os
import pickle

if not spacy.util.is_package("en_core_web_sm"):
    import spacy.cli
    spacy.cli.download('en_core_web_sm')

TEXT = data.Field(sequential=True, tokenize='spacy', tokenizer_language='en_core_web_sm', batch_first=True, lower=True, use_vocab=True, include_lengths=False ,fix_length=300)
LABEL = data.Field(sequential=False, use_vocab=True, pad_token=None, unk_token=None, batch_first=True)

train_data, valid_data, test_data = datasets.nli.SNLI.splits(TEXT, LABEL, root = 'data')

# we build the vocab if does not exit
if os.path.exists("vocab.pkl"):
    print("Loading vocabulary...")
    with open("vocab.pkl", 'rb') as f:
        TEXT.vocab = pickle.load(f)
else:
    print("Building vocabulary...")
    TEXT.build_vocab(train_data, vectors=vocab.GloVe(name='840B', dim=300))
    with open("vocab.pkl", 'wb') as f:
        pickle.dump(TEXT.vocab, f)
LABEL.build_vocab(train_data)
VOCABULARY = TEXT.vocab

print('Data loaded successfully!')