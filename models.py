import numpy as np
import torch 
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from collections import OrderedDict
from utils import TEXT


class AWE_Encoder(nn.Module):
    def __init__(self, embeddings:torch.tensor):
        super(AWE_Encoder, self).__init__()
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)
    
    def forward(self, input:torch.tensor) -> torch.tensor :
        embeds = self.embedding_layer(input)
        output = torch.mean(embeds)
        return output
    

class Unidirectional_LSTM(nn.Module):
    def __init__(self, embeddings:torch.tensor, hidden_dim:int):
        super(Unidirectional_LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.LSTM = nn.LSTM(300, hidden_dim, 1, bidirectional=False)
    
    def forward(self, input:torch.tensor) -> torch.tensor :
        embeds = self.embedding_layer(input)
        output, (hidden_states, cell_states) = self.LSTM(embeds)
        return hidden_states


class Bidirectional_LSTM(nn.Module):
    def __init__(self, embeddings:torch.tensor, hidden_dim:int):
        super(Bidirectional_LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.LSTM = nn.LSTM(300, hidden_dim, 1, bidirectional=True)
    
    def forward(self, input:torch.tensor) -> torch.tensor :
        embeds = self.embedding_layer(input)
        output, (hidden_states, cell_states) = self.LSTM(embeds)
        return hidden_states
    

class MaxPool_Bidirectional_LSTM(nn.Module):
    def __init__(self, embeddings:torch.tensor, hidden_dim:int):
        super(MaxPool_Bidirectional_LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.LSTM = nn.LSTM(300, hidden_dim, 1, bidirectional=True)
    
    def forward(self, input:torch.tensor) -> torch.tensor :
        embeds = self.embedding_layer(input)
        output, (hidden_states, cell_states) = self.LSTM(embeds)
        out_hidden_states = hidden_states.max(1)#.values
        return out_hidden_states
       

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

    def forward(self, input:torch.tensor)-> torch.tensor:
        out = self.net(input)
        return out



class TotalNN(pl.LightningModule):
    def __init__(self, encoder_type:str):
        super(TotalNN, self).__init__()
        
        self.save_hyperparameters()

        if encoder_type == "awe":
            self.encoder = AWE_Encoder(TEXT.vocab.vectors)
            self.net = Classifier(encoder_type, None)
        elif encoder_type == "uni_lstm":
            self.encoder = Unidirectional_LSTM(TEXT.vocab.vectors,2048)
            self.net = Classifier(encoder_type, SENTENCE_DIM)
        elif encoder_type == "bi_lstm":
            self.encoder = Bidirectional_LSTM(TEXT.vocab.vectors,2048)
            self.net = Classifier(encoder_type, SENTENCE_DIM)
        elif encoder_type == "pooled_bi_lstm":
            self.encoder == MaxPool_Bidirectional_LSTM(TEXT.vocab.vectors,2048)
            self.net = Classifier(encoder_type, SENTENCE_DIM)

        self.criterion = nn.CrossEntropyLoss() # as long as the loss module is cross entropy, we don't need to add softmax 


    def forward(self, premise:torch.tensor, hypothesis:torch.tensor) -> torch.tensor:
        u = self.encoder(premise)
        v = self.encoder(hypothesis)
        classifier_input = torch.cat([premise, hypothesis, torch.abs(u-v), u*v], 1)
        pred = self.net(classifier_input)
        return pred
    
    def configure_optimizers(self):
        # Create Optimizer
        #optimizer = torch.optim.Adam(self.parameters(), lr=0.02)

        optimizer = torch.optim.Adam([{'params': self.encoder.parameters()},
                {'params': self.net.parameters()}], lr=0.02)
        
        scheduler = {
            'scheduler': StepLR(optimizer=optimizer, step_size=1, gamma=0.99),
            'name': 'lr'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        premise = batch.premise
        hypothesis = batch.hypothesis
        label = batch.label
        output = self.forward(premise,hypothesis)
        loss = self.criterion(output,label)
        acc = (output.argmax(dim=-1) == label).float().mean()
        self.log("train_acc", loss)
        self.log("train_loss", acc)
        train_tensorboard_logs = {"train_loss": loss, "train_acc": acc}
        return train_tensorboard_logs
    
    def validation_step(self, batch, batch_idx):
        premise = batch.premise
        hypothesis = batch.hypothesis
        label = batch.label
        output = self.forward(premise,hypothesis)
        loss = self.criterion(output,label)
        acc = (output.argmax(dim=-1) == label).float().mean()
        self.log("valid_acc", loss)
        self.log("valid_loss", acc)
        valid_tensorboard_logs = {"valid_loss": loss, "valid_acc": acc}
        return valid_tensorboard_logs

    def test_step(self, batch, batch_idx):
        premise = batch.premise
        hypothesis = batch.hypothesis
        label = batch.label
        output = self.forward(premise,hypothesis)
        loss = self.criterion(output,label)
        acc = (output.argmax(dim=-1) == label).float().mean()
        self.log("test_acc", loss)
        self.log("test_loss", acc)
        test_tensorboard_logs = {"test_loss": loss, "test_acc": acc}
        return test_tensorboard_logs




        
