from torchtext import data,datasets,vocab
import torch
import spacy
import os
import pickle

# Needed for the tokenization of the dataset
if not spacy.util.is_package("en_core_web_sm"):
    import spacy.cli
    spacy.cli.download('en_core_web_sm')

class SNLI_data():
    def __init__(self):

        # Define the fields for the dataset
        self.text = data.Field(
            sequential=True,
            tokenize='spacy',
            tokenizer_language='en_core_web_sm',
            batch_first=True,
            lower=True,
            use_vocab=True,
            include_lengths=True
        )
        self.label = data.Field(
            sequential=False,
            use_vocab=True,
            pad_token=None,
            unk_token=None,
            batch_first=True,
            is_target=True
        )
        
        # Load the pre-trained GloVe embeddings
        self.glove_embeds = vocab.Vectors(name='glove.840B.300d.txt', max_vectors=300)
        
        # Load the SNLI dataset
        self.train_data, self.valid_data, self.test_data = datasets.nli.SNLI.splits(
            self.text,
            self.label,
            root='data'
        )
        
        # Build the vocabulary using pre-trained embeddings
        if os.path.exists("vocab.pkl"):
            with open("vocab.pkl", 'rb') as f:
                self.text.vocab = pickle.load(f)
        else:
            self.text.build_vocab(self.train_data, self.valid_data, vectors = self.glove_embeds)
            with open("vocab.pkl", 'wb') as f:
                pickle.dump(self.text.vocab, f)

        self.label.build_vocab(self.train_data)
        self.text.vocab.vectors[self.text.vocab.stoi['<unk>']] = torch.mean(self.text.vocab.vectors, dim=0)
        self.text.vocab.vectors[self.text.vocab.stoi['<pad>']] = 0
        self.vocabulary = self.text.vocab


class SNLI_Dataloader():
    def __init__(self, iterator):
        # Takes an iterable
        self.iterator = iterator

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        # Returns the data in form we want them
        for batch in self.iterator:
            premise = getattr(batch, 'premise')
            hypothesis = getattr(batch, 'hypothesis')
            label = getattr(batch, 'label')
            yield premise, hypothesis, label

if __name__ == '__main__':
    # We do this in order to create the .vector_cache dir which is needed for the other scripts
    dummy_data = SNLI_data()