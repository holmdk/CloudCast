import torch
from argparse import Namespace
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from collections import OrderedDict
import pytorch_lightning as pl

from CloudCast.src.data.CTDataLoader import CloudDataset
from CloudCast.src.data.preprocess_data import pre_process_data


from CloudCast.src.models.convlstm_autoencoder import DeepAutoencoderConvLSTM

# from convlstm.src.models.DeepAE_ConvLSTM import DeepAutoencoderConvLSTM
import os
from pytorch_lightning import Trainer


# import matplotlib.pyplot as plt


##########################
### MODEL
##########################

class ConvLSTMModel(pl.LightningModule):
    def __init__(self, hparams=None, path="/media/data/xarray/"):
        super(ConvLSTMModel, self).__init__()
        self.hparams = hparams

        self.nf = hparams.nf
        self.batch_size = hparams.batch_size # ['batch_size']  # 8
        self.num_classes = hparams.num_classes # ['num_classes'] #  4
        self.num_channels_in = hparams.num_channels_in  # ['num_channels_in']  # 4
        self.normalize = hparams.normalize  # ['normalize'] # False

        self.path = path

        self.model = DeepAutoencoderConvLSTM(nf=self.nf, in_chan=self.num_channels_in,
                                             n_classes=self.num_classes)#.cuda()

    def forward(self, x):
        x = x.unsqueeze(2)
        # x = x.to(device='cuda')

        output = self.model(x, future_seq=15)
        probas = F.softmax(output, dim=1)
        return probas

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat[:, :, -16:, :, :]
        loss = F.cross_entropy(y_hat, y.long())

        y_hat_class = torch.argmax(y_hat, 1)
        hits = y_hat_class == y
        accuracy = np.mean(hits.detach().cpu().numpy())
        accuracy = torch.tensor(accuracy)#.cuda()
        tensorboard_logs = {'train_loss': loss, 'accuracy': accuracy}

        return {'loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y, timestamp = batch
        y_hat = self.forward(x)
        y_hat = y_hat[:, :, -16:, :, :]
        loss = F.cross_entropy(y_hat, y.long())

        y_hat_class = torch.argmax(y_hat, 1)
        hits = y_hat_class == y
        accuracy = np.mean(hits.detach().cpu().numpy())
        accuracy = torch.tensor(accuracy)#.cuda()

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
        return torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.9, 0.98), eps=1e-9)

    def train_dataloader(self):
        # Initialize data loader
        train_set = CloudDataset(nc_filename=self.path + 'TrainCloud.nc',
                                 root_dir=self.path,
                                 sequence_start_mode='unique',
                                 n_lags=16,
                                 transform=pre_process_data,
                                 normalize=self.normalize,
                                 nan_to_zero=True,
                                 restrict_classes=True,
                                 return_timestamp=False,
                                 frames=32)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True)

        return train_loader

    def test_dataloader(self):
        test_set = CloudDataset(nc_filename=self.path + 'TestCloud.nc',
                                root_dir=self.path,
                                sequence_start_mode='unique',
                                n_lags=16,
                                transform=pre_process_data,
                                normalize=self.normalize,
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
    trainer = Trainer(max_epochs=config['max_epochs'], gpus=config['num_gpus'], distributed_backend=config['backend'],
                      early_stop_callback=False)
    trainer.fit(model)

    trainer.save_checkpoint(os.getcwd() + '/checkpoints/ae-convlstm.ckpt')


def resume_training(model, config):
    trainer = Trainer(max_epochs=config['max_epochs'], gpus=config['num_gpus'], distributed_backend=config['backend'],
                      early_stop_callback=False, resume_from_checkpoint=os.getcwd() + '/checkpoints/ae-convlstm.ckpt')
    trainer.fit(model)


def run_evaluation(model, config):
    trainer = Trainer(gpus=config['num_gpus'], distributed_backend=config['backend'], early_stop_callback=False,
                      resume_from_checkpoint=os.getcwd() + '/checkpoints/ae-convlstm.ckpt')
    model.freeze()
    preds = trainer.test(model)

    return preds



# def retrieve_predictions(model):
#     trainer = Trainer(gpus=2, distributed_backend='dp', early_stop_callback=False,
#                       resume_from_checkpoint='/home/local/DAC/ahn/Documents/dcwis.convlstm/model_checkpoints/model.ckpt')
#     model.freeze()
#
#     test_loader = model.test_dataloader()[0]
#
#     for i, (imgs_X, imgs_Y, time_stamp) in enumerate(test_loader):
#         pred = model.forward(imgs_X)
#         pred = pred[:, :, -16:, :, :]
#         y_true = imgs_Y.cuda().long()
#
#         loss = F.cross_entropy(pred, y_true)
#
#         print(loss)
#
#         pred_label = torch.argmax(pred, dim=1)
#         plt.imshow(y_true[0, 0, :, :].detach().cpu())



if __name__ == '__main__':
    # argparse with arguments ideally
    general_config = {
        'num_gpus': 1,
        'backend': 'dp',
        'max_epochs': 500
    }

    hparams = Namespace(
        nf=64,
        batch_size=8,
        num_classes=4,
        num_channels_in=4,  # the different cloud types
        normalize=False
    )
    model = ConvLSTMModel(hparams)
    train_model(model, general_config)
    # preds = run_evaluation()
