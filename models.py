import torch 
from torch import nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict
from utils import TEXT


class AWE_Encoder(nn.Module):
    def __init__(self, embeddings:torch.tensor):
        super(AWE_Encoder, self).__init__()
        
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=1)
    
    def forward(self, input:torch.tensor) -> torch.tensor :
        embeds = self.embedding_layer(input)
        output = torch.mean(embeds, dim=1)
        return output
    

class Unidirectional_LSTM(nn.Module):
    def __init__(self, embeddings:torch.tensor):
        super(Unidirectional_LSTM, self).__init__()

        self.sentence_dim = 2048
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=1)
        self.LSTM = nn.LSTM(input_size=300,hidden_size=self.sentence_dim, num_layers=1, batch_first=True, bidirectional=False)
    
    def forward(self, input:torch.tensor) -> torch.tensor :
        embeds = self.embedding_layer(input)
        output, (hidden_states, cell_states) = self.LSTM(embeds)
        hidden_states = torch.reshape(hidden_states, (-1,self.sentence_dim))
        return hidden_states


class Bidirectional_LSTM(nn.Module):
    def __init__(self, embeddings:torch.tensor):
        super(Bidirectional_LSTM, self).__init__()

        self.sentence_dim = 2048
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=1)
        self.LSTM = nn.LSTM(input_size=300,hidden_size=self.sentence_dim, num_layers=1, batch_first=True, bidirectional=True)
    
    def forward(self, input:torch.tensor) -> torch.tensor :
        embeds = self.embedding_layer(input)
        output, (hidden_states, cell_states) = self.LSTM(embeds)
        hidden_states = torch.reshape(hidden_states, (2,-1,self.sentence_dim))
        hiddenstates = torch.cat([hidden_states[0], hidden_states[1]],1)
        return hiddenstates
    

class MaxPool_Bidirectional_LSTM(nn.Module):
    def __init__(self, embeddings:torch.tensor):
        super(MaxPool_Bidirectional_LSTM, self).__init__()

        self.sentence_dim = 4096
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=1)
        self.LSTM = nn.LSTM(input_size=300,hidden_size=self.sentence_dim, num_layers=1, batch_first=True, bidirectional=True)
    
    def forward(self, input:torch.tensor) -> torch.tensor :
        embeds = self.embedding_layer(input)
        output, (hidden_states, cell_states) = self.LSTM(embeds)
        out_hidden_states = torch.max(hidden_states,dim=1).values #hidden_states.max(1).values
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

    def forward(self, encoded_embeds:torch.tensor)-> torch.tensor:
        out = self.net(encoded_embeds)
        return out




class Enc_MLP(pl.LightningModule):
    def __init__(self, args):
        super(Enc_MLP, self).__init__()
        
        self.save_hyperparameters()
        self.args = args

        if args.encoder_type == "awe":
            self.encoder = AWE_Encoder(TEXT.vocab.vectors)
            self.net = Classifier("awe", 0)
        elif args.encoder_type == "uni_lstm":
            self.encoder = Unidirectional_LSTM(TEXT.vocab.vectors)
            self.net = Classifier("uni_lstm", self.encoder.sentence_dim)
        elif args.encoder_type == "bi_lstm":
            self.encoder = Bidirectional_LSTM(TEXT.vocab.vectors)
            self.net = Classifier("bi_lstm", self.encoder.sentence_dim)
        elif args.encoder_type == "pooled_bi_lstm":
            self.encoder == MaxPool_Bidirectional_LSTM(TEXT.vocab.vectors)
            self.net = Classifier("pooled_bi_lstm", self.encoder.sentence_dim)

        self.criterion = nn.CrossEntropyLoss() # as long as the loss module is cross entropy, we don't need to add softmax 
        self.valid_accuracy = []

    def forward(self, premise:torch.tensor, hypothesis:torch.tensor) -> torch.tensor:
        u = self.encoder(premise)
        v = self.encoder(hypothesis)
        classifier_input = torch.cat([u, v, torch.abs(u-v), u*v], 1)
        pred = self.net(classifier_input)
        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD([{'params': self.encoder.parameters()},
                {'params': self.net.parameters()}], lr = self.args.lr)
        
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
        self.log("train_acc", acc)
        self.log("train_loss", loss)
        train_tensorboard_logs = {"loss": loss, "acc": acc}
        return train_tensorboard_logs
    
    def on_train_epoch_end(self):
        for pg in self.trainer.optimizers[0].param_groups:
            current_lr = pg['lr']
            if current_lr < 10e-5: 
                self.trainer.should_stop = True #return -1


    def validation_step(self, batch, batch_idx):
        premise = batch.premise
        hypothesis = batch.hypothesis
        label = batch.label
        output = self.forward(premise,hypothesis)
        loss = self.criterion(output,label)
        acc = (output.argmax(dim=-1) == label).float().mean()
        self.valid_accuracy.append(acc)
        self.log("val_acc", acc)
        self.log("val_loss", loss)
        #valid_tensorboard_logs = {"valid_loss": loss, "valid_acc": acc}
        return acc
    
    def on_validation_epoch_end(self):
        #self.last_val_acc = self.current_acc
        #self.current_acc = sum(acc) / len(acc)
        if self.valid_accuracy[-1] < self.valid_accuracy[-2]:
            for pg in self.trainer.optimizers[0].param_groups:
                pg['lr'] *= 0.2


    def test_step(self, batch, batch_idx):
        premise = batch.premise
        hypothesis = batch.hypothesis
        label = batch.label
        output = self.forward(premise,hypothesis)
        loss = self.criterion(output,label)
        acc = (output.argmax(dim=-1) == label).float().mean()
        self.log("test_acc", acc)
        self.log("test_loss", loss)
        #test_tensorboard_logs = {"test_loss": loss, "test_acc": acc}
        return acc
    
