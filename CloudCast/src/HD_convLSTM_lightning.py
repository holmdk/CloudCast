import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from collections import OrderedDict
import pytorch_lightning as pl

from torchvision.datasets import MNIST

# from torchvision.models.resnet import ResNet, BasicBlock, resnet50

from dcwis.satelliteloader.CTDataLoader import CloudDataset
from dcwis.satelliteloader.utils.pre_process import pre_process_data

from convlstm.src.models.DeepAE_ConvLSTM import DeepAutoencoderConvLSTM
import os
from pytorch_lightning import Trainer
# import matplotlib.pyplot as plt


##########################
### MODEL
##########################

class ConvLSTMModel(pl.LightningModule):

    def __init__(self, hparams=None):
        super(ConvLSTMModel, self).__init__()
        self.hparams = hparams

        self.nf = 64
        self.batch_size = 8
        self.num_classes = 4
        self.num_channels_in = 4
        self.normalize = False

        # self.path = "/media/oldL/data"
        self.path = "/data/"

        self.model = DeepAutoencoderConvLSTM(nf=self.nf, in_chan=self.num_channels_in,
                                             n_classes=self.num_classes).cuda()  # .half()

    def forward(self, x):
        x = x.unsqueeze(2)
        # x = x.cuda()
        x = x.to(device='cuda')

        output = self.model(x, future_seq=15)
        probas = F.softmax(output, dim=1)
        return probas

    def training_step(self, batch, batch_idx):
        # REQUIRED
        # self.zero_grad()
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat[:, :, -16:, :, :]
        loss = F.cross_entropy(y_hat, y.long())

        y_hat_class = torch.argmax(y_hat, 1)
        hits = y_hat_class == y
        accuracy = np.mean(hits.detach().cpu().numpy())
        accuracy = torch.tensor(accuracy).cuda()
        # tensorboard_logs = {'train_loss': loss}
        tensorboard_logs = {'train_loss': loss, 'accuracy': accuracy}

        return {'loss': loss, 'log': tensorboard_logs}


    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y, timestamp = batch
        y_hat = self.forward(x)
        y_hat = y_hat[:, :, -16:, :, :]
        loss = F.cross_entropy(y_hat, y.long())

        y_hat_class = torch.argmax(y_hat, 1)
        hits = y_hat_class == y
        accuracy = np.mean(hits.detach().cpu().numpy())
        accuracy = torch.tensor(accuracy).cuda()

        output = OrderedDict({
            'loss': loss,
            'accuracy': accuracy
        })

        return output

        # return {'test_loss': loss,
        #         'accuracy': np.mean(hits.detach().cpu().numpy())}


    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_accuracy = torch.stack([x['accuracy'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss, 'accuracy': avg_accuracy}
        return {'avg_test_loss': avg_loss, 'avg_accuracy': avg_accuracy, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=2e-4, betas=(0.9, 0.98), eps=1e-9)
        # return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.5)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        # Initialize data loader
        train_set = CloudDataset(nc_filename=self.path + '/data_sets/train/ct_and_sunhours_train.nc',  # TrainCloud
                                 root_dir=self.path + '/data_sets',
                                 sequence_start_mode='unique',
                                 n_lags=16,
                                 transform=pre_process_data,
                                 normalize=self.normalize,
                                 nan_to_zero=True,
                                 restrict_classes=True,
                                 return_timestamp=False,
                                 run_tests=False,
                                 day_only=False,  # Tr
                                 include_metachannels=False,
                                 frames=32)
        # find out how much is loaded into memory!!
        # february looks better than Jan! Loss works better

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            # num_workers=8,
            shuffle=True)

        # valid_iter = iter(train_loader)
        # val_gt, val_gt_y = valid_iter.next()

        return train_loader

    @pl.data_loader
    def test_dataloader(self):
        test_set = CloudDataset(nc_filename=self.path + '/data_sets/test/ct_and_sunhours_test.nc',
                                # '/2018M1_CT.nc'  # '/TestCloudHourly.nc'
                                root_dir=self.path,
                                sequence_start_mode='unique',
                                n_lags=16,
                                transform=pre_process_data,
                                normalize=self.normalize,
                                nan_to_zero=True,
                                run_tests=False,
                                return_timestamp=True,
                                restrict_classes=True,
                                day_only=False,  # Tr
                                include_metachannels=False,
                                frames=32)

        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=self.batch_size,
            # num_workers=1,
            shuffle=False
        )
        return test_loader



model = ConvLSTMModel()



def train_model():
    trainer = Trainer(max_epochs=500, gpus=2, distributed_backend='dp', early_stop_callback=False)
    trainer.fit(model)

    trainer.save_checkpoint(os.getcwd() + '/model_checkpoints/hd_model.ckpt')


def resume_training():
    trainer = Trainer(max_epochs=500, gpus=2, distributed_backend='dp', early_stop_callback=False,
                      resume_from_checkpoint='/home/local/DAC/ahn/Documents/dcwis.convlstm/model_checkpoints/hd_model.ckpt')
    trainer.fit(model)


def run_evaluation():
    trainer = Trainer(gpus=2, distributed_backend='dp', early_stop_callback=False,
                      resume_from_checkpoint='/home/local/DAC/ahn/Documents/dcwis.convlstm/model_checkpoints/hd_model.ckpt')
    model.freeze()
    preds = trainer.test(model)

    return preds




def retrieve_predictions():
    trainer = Trainer(gpus=2, distributed_backend='dp', early_stop_callback=False,
                      resume_from_checkpoint='/home/local/DAC/ahn/Documents/dcwis.convlstm/model_checkpoints/hd_model.ckpt')
    model.freeze()

    test_loader = model.test_dataloader()[0]

    for i, (imgs_X, imgs_Y, time_stamp) in enumerate(test_loader):
        pred = model.forward(imgs_X)
        pred = pred[:, :, -16:, :, :]
        y_true = imgs_Y.cuda().long()

        loss = F.cross_entropy(pred, y_true)

        print(loss)

        pred_label = torch.argmax(pred, dim=1)
        plt.imshow(y_true[0, 0, :, :].detach().cpu())






if __name__ == '__main__':
    train_model()
    # preds = run_evaluation()









