import sys 
import io
import numpy as np
import torch
import logging
import argparse
import spacy
from models import Enc_MLP
from utils import SNLI_data

PATH_SENTEVAL = './SentEval/'
PATH_DATA = './SentEval/data'
PATH_GLOVE = './.vector_cache/glove.840B.300d.txt'

sys.path.insert(0, PATH_SENTEVAL)
import senteval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The following function was taken from SentEval/senteval/utils.py and was 
# written by the research team working on the SentEval project
def create_dictionary(sentences):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1
    newwords = {}
    for word in words:
        newwords[word] = words[word]
    words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2
    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# The following function was taken from SentEval/senteval/bow.py and was 
# written by the research team working on the SentEval project
def get_wordvec(path_to_vec, word2id):
    word_vec = {}
    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


# SentEval prepare and batcher
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_GLOVE, params.word2id)
    params.wvec_dim = 300
    return 


def batcher(params, batch):
    # Get the word vectors for the words in the sentences
    embeddings = []
    sentence_lengths = []
    word2vec = params.word_vec
    wvec_dim = params.wvec_dim

    for sentence in batch:
        sent_representation = []
        for word in sentence:
            word_vector = word2vec.get(word, None)
            if word_vector is not None:
                sent_representation.append(word_vector)
        if not sent_representation:
            word_vector = np.zeros((1, wvec_dim))
            sent_representation.append(word_vector)
        sent_representation = np.vstack(sent_representation)
        sentence_lengths.append(sent_representation.shape[0])
        embeddings.append(sent_representation)

    # Pad the sequence
    max_length = max(sentence_lengths)
    padded_embeddings = np.zeros((len(batch), max_length, wvec_dim))
    for i, sent_emb in enumerate(embeddings):
        padded_embeddings[i, :sent_emb.shape[0], :] = sent_emb

    # Convert to PyTorch tensor
    padded_embeddings = torch.from_numpy(padded_embeddings).float()
    # Extract the length because it is needed for the forward function of the encoder
    sentence_lengths = torch.tensor(sentence_lengths)

    # Pass through the model
    with torch.no_grad():
        padded_embeddings = padded_embeddings.to(device)
        sentence_lengths = sentence_lengths.to(device)
        encoded_embeddings = model.encoder(padded_embeddings, sentence_lengths)

    return encoded_embeddings.cpu().detach().numpy()


# SentEval params
params_senteval = {'task_path': PATH_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":

    parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--encoder_type', default='awe', type=str,
                        help='What model to use.',
                        choices=['awe', 'uni_lstm', 'bi_lstm', 'pooled_bi_lstm'])
    args = parser.parse_args()
    snli_data = SNLI_data()
    vocab_vectors = snli_data.vocabulary.vectors
    model = Enc_MLP(vocab_vectors,args).to(device)
    model.eval()

    # Load from checkpoint depending on the encoder type
    if (args.encoder_type == 'awe'):
        checkpoint = torch.load('checkpoints/awe/best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict']) 
        
    elif (args.encoder_type == 'uni_lstm'):
        checkpoint = torch.load('checkpoints/uni_lstm/best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict']) 
    
    elif (args.encoder_type == 'bi_lstm'):
        checkpoint = torch.load('checkpoints/bi_lstm/best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])       

    elif (args.encoder_type == 'pooled_bi_lstm'):
        checkpoint = torch.load('checkpoints/pooled_bi_lstm/best_model.pt')
        model.load_state_dict(checkpoint['model_state_dict']) 
    
    else:
        raise TypeError("Invalid encoder type! Choices=['awe', 'uni_lstm', 'bi_lstm', 'pooled_bi_lstm']")

    # Run the senteval
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC', 'SICKRelatedness', 'SICKEntailment', 'STS14']
    results = se.eval(transfer_tasks)

    # Save the results
    torch.save(results, args.encoder_type + "_SentEvalResults.pt")

    # Print the results
    print(results)