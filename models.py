import torch 
from torch import nn
from collections import OrderedDict


class AWE_Encoder(nn.Module):
    def __init__(self):
        super(AWE_Encoder, self).__init__() 
        # The init function is empty as the average of the embeddings is done in the forward function
            
    def forward(self, embeddings:torch.tensor, sentence_length:torch.tensor) -> torch.tensor :
        # Computes the AWE
        output = torch.sum(embeddings, dim=1) / torch.unsqueeze(sentence_length,dim=1).float()
        return output
    

class Unidirectional_LSTM(nn.Module):
    def __init__(self):
        super(Unidirectional_LSTM, self).__init__()
        # Single layer LSTM with input size = 300 and hidden size = 2048
        # batch_first = True, as it depends on how you load the data from the dataloader
        self.lstm = nn.LSTM(input_size=300, hidden_size=2048 , num_layers=1, batch_first=True, bidirectional=False)
    
    def forward(self, embeddings:torch.tensor, sentence_length:torch.tensor) -> torch.tensor :

        # Sort the lengths by descending order
        sorted_lengths, sorted_indices = torch.sort(sentence_length, descending = True)
        # Choose the embeddings based on the sort done above
        sorted_embeddings = torch.index_select(embeddings, dim=0, index=sorted_indices)
        # Pack the padded sequence, sorted_lengths has to be on CPU, as it causes conflicts if on gpu
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(sorted_embeddings, sorted_lengths.to('cpu'), batch_first = True)
        # nn.LSTM returns: (ALL, (h0, c0)), where ALL = the hidden states and (h0, c0) = hidden state and cell state of the last time-step
        hidden_states = self.lstm(packed_embeddings)[1][0].squeeze(0) # Initial (h_0, c_0) are zeros if not provided.
        # No need to unpack, because we keep the hidden states 
        # Basically the opposite of what we did before
        _, unsorted_indices = torch.sort(sorted_indices)
        output = torch.index_select(hidden_states, dim=0, index=unsorted_indices)
        return output


class Bidirectional_LSTM(nn.Module):
    def __init__(self):
        super(Bidirectional_LSTM, self).__init__()
        # Bidirectional = True to take the forward and the reverse state
        self.lstm = nn.LSTM(input_size=300, hidden_size=2048, num_layers=1, batch_first=True, bidirectional=True)
    
    def forward(self, embeddings:torch.tensor, sentence_length:torch.tensor) -> torch.tensor :

        sorted_lengths, sorted_indices = torch.sort(sentence_length, descending =True)
        sorted_embeddings = torch.index_select(embeddings, dim=0, index=sorted_indices)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(sorted_embeddings, sorted_lengths.to('cpu'), batch_first = True)
        # We are interested in the hidden states
        _, (hidden_states, _) = self.lstm(packed_embeddings)
        # We concatenate the forward and the reverse state H_out = [H_forward H_reverse]
        final_states = torch.hstack((hidden_states[0],hidden_states[1]))
        _, unsorted_indices = torch.sort(sorted_indices)
        output = torch.index_select(final_states, dim=0, index=unsorted_indices)
    
        return output
    

class MaxPool_Bidirectional_LSTM(nn.Module):
    def __init__(self):
        super(MaxPool_Bidirectional_LSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=300, hidden_size=2048, num_layers=1, batch_first=True, bidirectional=True)
    
    def forward(self, embeddings:torch.tensor, sentence_length:torch.tensor) -> torch.tensor :

        sorted_lengths, sorted_indices = torch.sort(sentence_length, descending =True)
        sorted_embeddings = torch.index_select(embeddings, dim=0, index=sorted_indices)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(sorted_embeddings, sorted_lengths.to('cpu'), batch_first = True)
        all_states, _ = self.lstm(packed_embeddings)
        # We pad the packed sequence, but instead of zeros, we pad with -10 so it won't cause any problems during max pooling
        hidden_states = nn.utils.rnn.pad_packed_sequence(all_states, batch_first = True,padding_value=-10)[0]
        _, unsorted_indices = torch.sort(sorted_indices)
        out = torch.index_select(hidden_states, dim=0, index=unsorted_indices)
        # Max pooling
        sorted_lengths = sorted_lengths.int()
        emb_out = [out[i][:sorted_lengths[i]] for i in range(len(out))]
        emb_max = [emb_out[i].max(dim=0)[0] for i in range(len(emb_out))]
        output = torch.vstack(emb_max)

        return output
       

class Classifier(nn.Module):
    def __init__(self, encoder_type:str, sentence_dim:int):
        super(Classifier, self).__init__()

        if encoder_type == "awe":
            self.net = nn.Sequential(OrderedDict([
                ("in", nn.Linear(4*300, 512)),
                ("act1", nn.Tanh()),
                ("hidden", nn.Linear(512,512)),
                ("act2", nn.Tanh()),
                ("out", nn.Linear(512,3))
            ]))
        else:
            self.net = nn.Sequential(OrderedDict([
                ("in", nn.Linear(4*sentence_dim, 512)),
                ("act1", nn.Tanh()),
                ("hidden", nn.Linear(512,512)),
                ("act2", nn.Tanh()),
                ("out", nn.Linear(512,3))
            ]))

    def forward(self, encoded_embeds:torch.tensor) -> torch.tensor:
        out = self.net(encoded_embeds)
        return out


class Enc_MLP(nn.Module):
    def __init__(self, weight, args):
        super(Enc_MLP, self).__init__()
        
        self.args = args
        # Embedding layer so the sentences are converted into embeddings
        # We set freeze = True, because we use the pretrained GloVE embeddings
        self.embedding_layer = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=1)

        if args.encoder_type == "awe":
            self.encoder = AWE_Encoder()
            self.net = Classifier("awe", 0)
        elif args.encoder_type == "uni_lstm":
            self.encoder = Unidirectional_LSTM()
            self.net = Classifier("uni_lstm", 2048)
        elif args.encoder_type == "bi_lstm":
            self.encoder = Bidirectional_LSTM()
            self.net = Classifier("bi_lstm", 2 * 2048) #4096 because we have bidirectional
        elif args.encoder_type == "pooled_bi_lstm":
            self.encoder = MaxPool_Bidirectional_LSTM()
            self.net = Classifier("pooled_bi_lstm", 2 * 2048) #same as above

    def forward(self, premise:torch.tensor, premise_length:torch.tensor, hypothesis:torch.tensor, hypothesis_length:torch.tensor) -> torch.tensor:
        premise = self.embedding_layer(premise)
        hypothesis = self.embedding_layer(hypothesis)    
        u = self.encoder(premise, premise_length)
        v = self.encoder(hypothesis, hypothesis_length)
        classifier_input = torch.cat([u, v, torch.abs(u-v), u*v], 1)
        output = self.net(classifier_input)
        return output
