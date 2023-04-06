from torchtext import data,datasets,vocab
import spacy
import os
import pickle

# spacy is needed in order to tokenize the corpus, so we download it
if not spacy.util.is_package("en_core_web_sm"):
    import spacy.cli
    spacy.cli.download('en_core_web_sm')

# we build the tokenized corpus
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm', batch_first=True)
LABEL = data.Field(sequential=False, is_target=True, unk_token=None)

train_data, valid_data, test_data = datasets.nli.SNLI.splits(TEXT, LABEL, root = './data')

# we build the vocab if does not exit
if not os.path.exists('./data/vocab.pkl'):
    TEXT.build_vocab(train_data, vectors = vocab.GloVe(name='840B', dim=300))
    with open("vocab.pkl", 'wb') as f:
        pickle.dump(TEXT.vocab, f)
else:
    with open("vocab.pkl", 'rb') as f:
        TEXT.vocab = pickle.load(f)
LABEL.build_vocab(train_data)

print('Data loaded successfully!')