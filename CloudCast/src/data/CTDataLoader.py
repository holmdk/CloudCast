"""
Data loading object
"""
# Author: Andreas Holm <ahn@eng.au.dk>
# Based upon https://github.com/weixiong-ur/mdgan/blob/master/video_folder_test.py

from torch.utils.data import Dataset
from CloudCast.src.data.preprocess_data import pre_process_data
import xarray as xr
import torch
import numpy as np

class CloudDataset(Dataset):
    """Cloud-labelled satellite images."""
    def __init__(self, nc_filename, root_dir, transform=None, frames=12, n_lags=1, restrict_classes=False,
                 normalize=True, nan_to_zero=True, return_timestamp=False, sequence_start_mode='unique'):

        # Load dataset into memory (do not do this for full dataset!
        self.cloud_frames = xr.open_dataarray(nc_filename).load()

        # config parameters
        self.root_dir = root_dir
        self.transform = transform
        self.frames = frames
        self.n_lags = n_lags
        self.restrict_classes = restrict_classes
        self.normalize = normalize
        self.nan_to_zero = nan_to_zero
        self.return_timestamp = return_timestamp
        self.sequence_start_mode = sequence_start_mode

        assert sequence_start_mode in ['all', 'optimal', 'unique'], 'sequence mode must be one of "all", "optimal" or "unique"'
        assert n_lags <= frames // 2, 'Mismatch between number of frames and lags, lags should not exceed 50% of n_frames'

        if self.sequence_start_mode == 'all':  # include all sequences despite potential overlaps
            self.possible_starts = np.array([i for i in range(len(self.cloud_frames.time) - self.frames)])
        elif self.sequence_start_mode == 'unique':  # do no allow overlaps
            # unique as ???
            self.possible_starts = np.array([i for i in range(len(self.cloud_frames.time) - self.frames)
                                             if i % self.n_lags == 0])
        else:
            # Optimal as we are not using all data (which gives too much emphasis ???? Shouldn't we just use all=
            pass


    def __len__(self):
        return len(self.possible_starts)
        # return len(self.cloud_frames.time)

    def __getitem__(self, idx):
        index = self.possible_starts[idx]

        clip = self.cloud_frames[:, :, index:index + self.frames]  # clip  # clip corresponds to an array of 32 frames

        if self.transform is not None:
            clip.values = self.transform(clip.values, nan_to_zero=self.nan_to_zero, normalize=self.normalize,
                                         restrict_classes_to_4=self.restrict_classes)

        # Save time stamps
        times = clip.time.values

        clip = np.array(clip)
        clip = torch.stack([torch.Tensor(i) for i in clip])  # transform into torch.tensors

        # Specify X and y according to number of lags
        y = clip[:, :, self.n_lags:]
        X = clip[:, :, 0:self.n_lags]

        y = y.permute(2, 0, 1)  # rearrange to have time steps, height and width as the shape
        X = X.permute(2, 0, 1)  # rearrange to have time steps, height and width as the shape

        if self.return_timestamp:
            ts = times.astype('datetime64[s]').astype('int')
            ts = torch.tensor(ts, dtype=torch.int64)

            return X, y, ts
        else:
            return X, y

if __name__ == '__main__':
    # Define root path
    # path = "/data/"
    path = '/media/data/xarray/'
    train_set = CloudDataset(nc_filename=path + 'TrainCloud.nc',  # =path + '/train/ct_and_sunhours_train.nc',
                             root_dir=path,
                             sequence_start_mode='unique',
                             n_lags=16,
                             transform=pre_process_data,
                             return_timestamp=True,
                             normalize=False,
                             nan_to_zero=True,
                             restrict_classes=True,
                             frames=32)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=2,
        shuffle=True)

    valid_iter = iter(train_loader)
    val_gt, val_gt_y, time_stamp = valid_iter.next()
    print('example video loaded.')
    time_stamp.numpy().astype('datetime64[s]')
