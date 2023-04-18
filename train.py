import argparse
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import BucketIterator
import torch.nn as nn
from models import *
from tqdm import tqdm
from utils import SNLI_data, SNLI_Dataloader


def evaluate_model(model, valid_dataloader, args):
    if torch.cuda.is_available():
        device = args.device
    else:
        device = 'cpu'
    eval_acc = 0
    eval_loss = 0
    model.eval()
    loss_module = nn.CrossEntropyLoss()

    with torch.no_grad():
        for premise, hypothesis, labels in tqdm(valid_dataloader):
            premise_sentence = premise[0].to(device)
            premise_length = premise[1].to(device)
            hypothesis_sentence = hypothesis[0].to(device)
            hypothesis_length = hypothesis[1].to(device)
            labels = labels.to(device)
            out = model(premise_sentence,premise_length, hypothesis_sentence, hypothesis_length)
            loss = loss_module(out,labels)
            eval_loss += loss.detach().item()
            preds = torch.argmax(out, dim=1)
            accuracy = torch.sum(preds == labels, dtype=torch.float32) / out.shape[0]
            eval_acc += accuracy
        eval_loss /= len(valid_dataloader)
        eval_acc /= len(valid_dataloader)
        
    return eval_loss, eval_acc



def run(args):

    # Seeds for reproduceable runs
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = args.device
    else:
        device = 'cpu'

    # Create the dataloaders
    snli_data = SNLI_data()
    train_iter, dev_iter, test_iter = BucketIterator.splits(datasets=(snli_data.train_data, snli_data.valid_data, snli_data.test_data),
                                                   batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
                                                   sort_key=lambda x: x.premise,
                                                   sort_within_batch=True,
                                                   device=device)
    train_loader = SNLI_Dataloader(train_iter)
    val_loader = SNLI_Dataloader(dev_iter)
    test_loader = SNLI_Dataloader(test_iter)

    # Initialize the model, optimizer and loss function
    model = Enc_MLP(snli_data.text.vocab.vectors,args).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)
    loss_module = nn.CrossEntropyLoss()

    # Lists needed for logging
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    best_val_acc = 0


    for epoch in range(args.max_epochs):
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for (premise, hypothesis, labels) in tqdm(train_loader):
            premise_sentence = premise[0].to(device)
            premise_length = premise[1].to(device)
            hypothesis_sentence = hypothesis[0].to(device)
            hypothesis_length = hypothesis[1].to(device)
            labels = labels.to(device)
            out = model(premise_sentence,premise_length, hypothesis_sentence, hypothesis_length)
            loss = loss_module(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = torch.argmax(out, dim=1)
            train_epoch_acc += torch.sum(preds == labels, dtype=torch.float32) / out.shape[0]
            train_epoch_loss += loss.detach().item()

        train_loss = train_epoch_loss/len(train_loader)
        train_loss_list.append(train_loss)
        train_acc = train_epoch_acc/len(train_loader)
        train_acc_list.append(train_acc)

        # Validation of the model for the current epoch
        val_loss, val_acc = evaluate_model(model, val_loader, args)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        print("Epoch :", epoch, "- train loss: ", train_loss)
        print("Epoch :", epoch, "- train acc: ", train_acc.item())
        print("Epoch :", epoch, "- val loss: ", val_loss)
        print("Epoch :", epoch, "- val acc: ", val_acc.item())

        # Tensorboard stuff
        writer.add_scalar('Train/epochs/loss', train_loss, epoch)
        writer.add_scalar('Train/epochs/accuracy', train_acc.item(), epoch)
        writer.add_scalar('Validation/epoch/loss', val_loss, epoch)
        writer.add_scalar('Validation/epoch/accuracy', val_acc.item(), epoch)

        epoch_lr = optimizer.param_groups[0]['lr']
        print("Current LR :", epoch_lr)

        # This will check if the folder checkpoints exists and will create it if it doesn't
        checkpoint_dir = 'checkpoints' 
        if not os.path.exists(os.path.join(checkpoint_dir, args.encoder_type)):
            os.makedirs(os.path.join(checkpoint_dir, args.encoder_type))

        # Save model checkpoints for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.checkpoint_path, args.encoder_type, args.saved_model_name))

            epoch_lr *= 0.99
            print("Multiplying LR by 0.99")
        else:
            epoch_lr *= 0.2
            print("Multiplying LR by 0.2")
        
        optimizer.param_groups[0]['lr'] = epoch_lr

        if optimizer.param_groups[0]['lr'] < 1e-5:
            print("Stop training")
            break
        
    # Load checkpoint
    model_file = os.path.join(args.checkpoint_path, args.encoder_type, args.saved_model_name)
    if os.path.isfile(model_file):
        checkpoint = torch.load(model_file)
        model.load_state_dict(checkpoint['model_state_dict']) 

    # Testing 
    test_loss, test_acc = evaluate_model(model,test_loader, args)
    print("Test accuracy :", test_acc.item())
    writer.add_scalar('Test_acc', test_acc.item())
    writer.add_scalar('Test_loss', test_loss)
    writer.close()
    return test_loss, test_acc.item()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--checkpoint_path', type = str, default = './checkpoints',
                          help='Directory of check point')
    parser.add_argument("--saved_model_name", type=str, default= 'best_model.pt',
                       help = 'Name of saved model')
    parser.add_argument('--encoder_type', type = str, default = 'pooled_bi_lstm',
                          help='Type of encoder (awe,uni_lstm,bi_lstm,pooled_bi_lstm)')
    parser.add_argument('--lr', type = float, default = 0.1,
                          help='Learning rate for training')
    parser.add_argument('--batch_size', type = int, default = 64,
                          help='batch size for training"')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for reproducibility')
    parser.add_argument('--max_epochs', type = int, default = 50,
                        help = 'Max epochs to train for')
    parser.add_argument('--device', type = str, default = 'cuda',
                        help = 'Device to run the model on')
    
    args = parser.parse_args()
    writer = SummaryWriter(os.path.join('tensorboard_logs', args.encoder_type))
    run(args)
