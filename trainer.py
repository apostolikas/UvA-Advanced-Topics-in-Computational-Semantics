import torch
import torchtext
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from utils import train_data, valid_data, test_data
from models import Enc_MLP
import os
import argparse


def train(args):

    os.makedirs(args.log_dir, exist_ok=True)

    train_dataloader = torchtext.data.BucketIterator(train_data, batch_size=args.batch_size, train=True)
    valid_dataloader = torchtext.data.BucketIterator(valid_data, batch_size=args.batch_size, train=True)
    test_dataloader = torchtext.data.BucketIterator(test_data, batch_size=args.batch_size, train=False)

    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs = 30,
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, monitor="val_acc", mode="max"),
                         enable_progress_bar=True)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    pl.seed_everything(args.seed)  # To be reproducible
    model = Enc_MLP(encoder_type=args.encoder_type)

    # Training
    trainer.fit(model,train_dataloader,valid_dataloader)

    # Testing
    model = Enc_MLP.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, dataloaders=test_dataloader, verbose=True)

    return test_result


if __name__ == "__main__":

    parser = argparse.ArguementParser(
        formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    # Model hyperparameters
    parser.add_argument('--encoder_type', default='awe', type=str,
                        help='Type of encoder (awe,uni_lstm,bi_lstm,pooled_bi_lstm)')

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.02, type=float,
                        help='Learning rate')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')

    # Other hyperparameters
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for reproducibility')
    parser.add_argument('--checkpoint_dir', default=None, type=str,
                        help='Directory of model checkpoint')
    parser.add_argument('--log_dir', default='model_logs', type=str,
                        help='Directory of PL logs')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    train(args)
