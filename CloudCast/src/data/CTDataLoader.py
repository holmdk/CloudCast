"""
Data loading object
"""
# Author: Andreas Holm <ahn@eng.au.dk>
# Based upon https://github.com/weixiong-ur/mdgan/blob/master/video_folder_test.py

from torch.utils.data import Dataset
# from dcwis.satelliteloader.CTDataValidator import DataValidator
from CloudCast.src.data.preprocess_data import pre_process_data
import xarray as xr
import torch
import numpy as np
import sys
import pandas as pd
import datetime as dt

class CloudDataset(Dataset):
    """Cloud-labelled satellite images."""

    def __init__(self, nc_filename, root_dir, transform=None, frames=12, n_lags=1, restrict_classes=False,
                 normalize=True, nan_to_zero=True, run_tests=True, return_timestamp=False, sequence_start_mode='unique',
                 include_metachannels=True, day_only=True):

        try:
            self.cloud_frames = xr.open_dataarray(nc_filename).load()
        except Exception as e:
            print('Could not load xarray filename %s as datarray due to %s, trying to open as dataset' % (nc_filename, e))

            try:
                self.cloud_frames = xr.open_dataset(nc_filename).load()
                if include_metachannels == False:
                    self.cloud_frames = self.cloud_frames.CT
            except Exception as e:
                print('Could not load xarray filename %s as dataset due to %s, exiting script' % (nc_filename, e))
                sys.exit()

        self.root_dir = root_dir
        self.transform = transform
        self.frames = frames
        self.n_lags = n_lags
        self.restrict_classes = restrict_classes
        self.normalize = normalize
        self.nan_to_zero = nan_to_zero
        # self.dataval = DataValidator(break_if_assertion=False)
        self.run_tests = run_tests
        self.return_timestamp = return_timestamp
        self.sequence_start_mode = sequence_start_mode
        self.include_metachannels = include_metachannels

        if day_only:
            sun_hrs = self.cloud_frames.sun_hours.load()
            summed_hours = sun_hrs.groupby('time').sum()
            expected_sun_hours = self.cloud_frames.lat.shape[0] * self.cloud_frames.lon.shape[0]
            self.cloud_frames = self.cloud_frames.where(summed_hours == expected_sun_hours, drop=True)

        assert sequence_start_mode in ['all', 'unique'], 'sequence mode must be one of "all" or "unique"'

        assert n_lags <= frames // 2, 'Mismatch between number of frames and lags, lags should not exceed 50% of n_frames'

        if self.sequence_start_mode == 'all':  # include all sequences despite potential overlaps
            self.possible_starts = np.array([i for i in range(len(self.cloud_frames.time) - self.frames)])
        else:
            # unique, there will only be potential overlap between train and test - but batches cannot be used
            # as training more than once per an epoch
            self.possible_starts = np.array([i for i in range(len(self.cloud_frames.time) - self.frames)
                                             if i % self.n_lags == 0])


    def __len__(self):
        return len(self.possible_starts)
        # return len(self.cloud_frames.time)

    def __getitem__(self, idx):
        index = self.possible_starts[idx]

        if self.include_metachannels:
            time_idx = self.cloud_frames.CT[:, :, index:index + self.frames].time
            clip = self.cloud_frames.sel(time=time_idx)

            if self.transform:
                clip.CT.values = self.transform(clip.CT.values, nan_to_zero=self.nan_to_zero, normalize=self.normalize,
                                                restrict_classes_to_4=self.restrict_classes)
                #TODO: We could also transform other variables, but not binary encoded ones

            clip = clip.to_array()

        else:
            clip = self.cloud_frames[:, :, index:index + self.frames]  # clip  # clip corresponds to an array of 32 frames
            if self.transform:
                clip.values = self.transform(clip.values, nan_to_zero=self.nan_to_zero, normalize=self.normalize,
                                             restrict_classes_to_4=self.restrict_classes)

            # Test to see if 15-minute granularity is specified correctly (time for 1000 runs is 0.83 seconds)
        if self.run_tests:
            self.dataval.granularity_test(clip)

        # Save time stamps
        times = clip.time.values

        clip = np.array(clip)
        clip = torch.stack([torch.Tensor(i) for i in clip])  # transform into torch.tensors

        if self.include_metachannels:
            y = clip[:, :, :, self.n_lags:]
            X = clip[:, :, :, 0:self.n_lags]

            y = y.permute(0, 3, 1, 2) #  rearrange to have channels, time steps, height and width as the shape
            X = X.permute(0, 3, 1, 2)

        else:
            y = clip[:, :, self.n_lags:]
            X = clip[:, :, 0:self.n_lags]

            # Test to see if shape mismatch between X and y (extremely fast, 1000 runs takes 0.00 seconds)
            if self.run_tests:
                self.dataval.lagging_rows_test(X, y)
                # Test to see if time differences between X and y is in accordance, 1000 runs takes 0.02 seconds
                self.dataval.lagging_time_differences_test(times, self.n_lags)

            y = y.permute(2, 0, 1)  # rearrange to have time steps, height and width as the shape
            X = X.permute(2, 0, 1)  # rearrange to have time steps, height and width as the shape


        if self.return_timestamp:
            ts = times.astype('datetime64[s]').astype('int')
            ts = torch.tensor(ts, dtype=torch.int64)

            return X, y, ts
        else:
            return X, y

def frequency_count(input):
    total = input.numel()
    class_0 = np.round((input == 0).sum().numpy() / total, 2)
    class_1 = np.round((input == 1).sum().numpy() / total, 2)
    class_2 = np.round((input == 2).sum().numpy() / total, 2)
    class_3 = np.round((input == 3).sum().numpy() / total, 2)

    freq = {
        'No Cloud': class_0,
        'Low Cloud': class_1,
        'Mid Cloud': class_2,
        'High Cloud': class_3
    }
    return freq


if __name__ == '__main__':
    # Define root path
    path = "/data/"
    train_set = CloudDataset(nc_filename=path + '/CloudFullHighRes.nc',  # =path + '/train/ct_and_sunhours_train.nc',
                             root_dir=path,
                             sequence_start_mode='unique',
                             n_lags=16,
                             transform=pre_process_data,
                             return_timestamp=True,
                             normalize=False,
                             nan_to_zero=True,
                             restrict_classes=True,
                             run_tests=False,
                             day_only=False,  # Tr
                             include_metachannels=False,
                             frames=32)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=2,
        shuffle=True)

    valid_iter = iter(train_loader)
    val_gt, val_gt_y, time_stamp = valid_iter.next()
    print('validation video loaded.')

    print(frequency_count(val_gt))

    time_stamp.numpy().astype('datetime64[s]')
