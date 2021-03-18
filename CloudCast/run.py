import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict

from src.data.CTDataLoader import CloudDataset
from src.data.preprocess_data import pre_process_data
from src.models.convlstm_autoencoder import DeepAutoencoderConvLSTM
from src.visualization.save_images_cartopy import save_images_cartopy

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


##########################
### MODEL
##########################

class ConvLSTMModel(pl.LightningModule):
    def __init__(self, hparams=None, path="/media/data/xarray/"):
        super(ConvLSTMModel, self).__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size
        self.num_classes = hparams.num_classes
        self.num_channels_in = hparams.num_channels_in
        self.future_steps = hparams.future_steps
        self.num_workers = hparams.num_workers

        self.path = path

        self.model = DeepAutoencoderConvLSTM(in_chan=self.num_channels_in,
                                             n_classes=self.num_classes)

        if hparams.loss == 'l1':
            raise NotImplementedError('L1 loss not implemented yet')
            # self.criterion = torch.nn.L1Loss()
        elif hparams.loss == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif hparams.loss == 'weighted_ce':
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([1 / 0.3, 1 / 0.3, 1 / 0.1, 1 / 0.3]),
                reduction='mean')
        elif hparams.loss == 'combined':
            raise NotImplementedError('combined loss not implemented yet')
            # self.a_loss = torch.nn.L1Loss()#.cuda()
            # self.b_loss = torch.nn.CrossEntropyLoss(
            #     weight=torch.tensor([1 / 0.3, 1 / 0.3, 1 / 0.1, 1 / 0.3]),
            #     reduction='mean')

    def forward(self, x):
        x = x.unsqueeze(2)
        # x = x.to(device='cuda')

        output = self.model(x, future_seq=self.future_steps - 1)
        probas = F.softmax(output, dim=1)
        return probas

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat[:, :, -self.future_steps:, :, :]

        if hparams.loss == 'combined':
            a_loss = self.a_loss(y_hat, y)
            b_loss = self.b_loss(y_hat, y)
            loss = a_loss + b_loss
        else:
            loss = self.criterion(y_hat, y.long())
            # loss = F.cross_entropy(y_hat, y.long())

        y_hat_class = torch.argmax(y_hat, 1)
        hits = y_hat_class == y
        accuracy = np.mean(hits.detach().cpu().numpy())
        accuracy = torch.tensor(accuracy)  # .cuda()
        tensorboard_logs = {'train_loss': loss, 'accuracy': accuracy}

        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y, timestamp = batch
        y_hat = self.forward(x)
        y_hat = y_hat[:, :, -self.future_steps:, :, :]
        loss = F.cross_entropy(y_hat, y.long())

        y_hat_class = torch.argmax(y_hat, 1)
        hits = y_hat_class == y
        accuracy = np.mean(hits.detach().cpu().numpy())
        accuracy = torch.tensor(accuracy)  # .cuda()

        output = OrderedDict({
            'loss': loss,
            'accuracy': accuracy
        })

        return output

    def test_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'accuracy': avg_accuracy}
        return {'avg_test_loss': avg_loss, 'avg_accuracy': avg_accuracy, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=hparams.lr, betas=(hparams.beta1, hparams.beta2),
                                     eps=hparams.eps)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # Initialize data loader
        train_set = CloudDataset(nc_filename=self.path + 'TrainCloud.nc',
                                 root_dir=self.path,
                                 sequence_start_mode='unique',
                                 n_lags=16,
                                 transform=pre_process_data,
                                 normalize=False,
                                 nan_to_zero=True,
                                 restrict_classes=True,
                                 return_timestamp=False,
                                 frames=32)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True)

        return train_loader

    def test_dataloader(self):
        test_set = CloudDataset(nc_filename=self.path + 'TestCloud.nc',
                                root_dir=self.path,
                                sequence_start_mode='unique',
                                n_lags=16,
                                transform=pre_process_data,
                                normalize=False,
                                nan_to_zero=True,
                                return_timestamp=True,
                                restrict_classes=True,
                                frames=32)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False
        )
        return test_loader


def train_model(model, config):
    trainer = Trainer(max_epochs=config.max_epochs, gpus=hparams.num_gpus, distributed_backend=hparams.data_backend,
                      early_stop_callback=False)
    trainer.fit(model)

    trainer.save_checkpoint(os.getcwd() + '/checkpoints/ae-convlstm.ckpt')


def resume_training(model, config):
    trainer = Trainer(max_epochs=config.max_epochs, gpus=hparams.num_gpus, distributed_backend=hparams.data_backend,
                      early_stop_callback=False, resume_from_checkpoint=hparams.pretrained_path)
    trainer.fit(model)


def run_evaluation(model, hparams, save_images_path):
    trainer = Trainer(gpus=hparams.num_gpus, distributed_backend=hparams.data_backend, early_stop_callback=False,
                      resume_from_checkpoint=hparams.pretrained_path)
    model.freeze()
    preds = trainer.test(model)

    save_predictions(trainer, model, save_images_path)

    return preds


def save_predictions(trainer, model, save_images_path):
    print('Creating predictions...')
    test_loader = trainer.test_dataloaders[0]

    for i, (imgs_X, imgs_Y, time_stamp) in enumerate(test_loader):
        if i > 4:
            sys.exit()  # we only run for 5 iterations

        ts = time_stamp.numpy()
        ts = ts.astype('datetime64[s]')
        target_times = ts[:, -16:]

        pred = model.forward(imgs_X.cuda())
        pred = pred[:, :, -16:, :, :]
        y_true = imgs_Y.cuda().long()

        loss = F.cross_entropy(pred, y_true)

        print(loss)

        pred_label = torch.argmax(pred, dim=1)
        save_images_cartopy(prediction_video=pred_label, ground_truth_video=y_true, target_times=target_times, batch=i,
                            lines=False, high_res_map=True, path=save_images_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str2bool, help="Train model from scratch or use pretrained model for inference")
    parser.add_argument("--lr", default=2e-4, type=float, help="The learning rate")
    parser.add_argument("--beta1", default=0.9, type=float, help="The learning rate")
    parser.add_argument("--beta2", default=0.98, type=float, help="The learning rate")
    parser.add_argument("--eps", default=1e-9, type=float, help="The learning rate")
    parser.add_argument("--loss", default='ce', type=str,
                        help="Loss function to use. Can be either ce, weighted_ce, l1, or combined")
    parser.add_argument("--batch_size", default=2, type=int, help="The batch size. Keep low if limited GPU memory")
    parser.add_argument("--num_gpus", default=1, type=int, help="How many GPU's to use")
    parser.add_argument("--data_backend", default='dp', type=str, help="The data backend. Can be either dp or ddp")
    parser.add_argument("--num_workers", default=4, type=int, help="How many data workers to spawn")
    parser.add_argument("--max_epochs", default=500, type=int, help="How many maximum epochs to run training")
    parser.add_argument("--num_classes", default=4, type=int, help="Number of target classes. Default is 4")
    parser.add_argument("--num_channels_in", default=4, type=int,
                        help="Number of input channels. Can be higher than num_classes if you use additional input channels")
    parser.add_argument("--future_steps", default=16, type=int, help="How many future time steps to predict")
    parser.add_argument("--pretrained_path", default='./models/pretrained.ckpt', type=str)

    hparams = parser.parse_args()

    print('\n')
    print('_________ Initializing ConvLSTM model _________')
    print('--------------------------------------------------------------------------------')
    print(hparams)
    print('--------------------------------------------------------------------------------')
    print('\n')

    model = ConvLSTMModel(hparams)

    if bool(hparams.train):
        print('Trainining model...')
        train_model(model, hparams)
    else:
        save_images_path = './saved_images/'
        preds = run_evaluation(model, hparams, save_images_path)
