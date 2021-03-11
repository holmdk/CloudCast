

import sys
sys.path.append('/home/local/DAC/ahn/Documents/dcwis.satelliteloader/dcwis/satelliteloader/')
from dcwis.satelliteloader.CTDataLoader import CloudDataset
from dcwis.satelliteloader.utils.pre_process import pre_process_data





# from dcwis.satelliteloader.utils.pre_process import pre_process_data
# from dcwis.satelliteloader.DataLoader import CloudDataset
import torch
import numpy as np
from torch.autograd import Variable
from convlstm.src.models.DeepAE_ConvLSTM import DeepAutoencoderConvLSTM
import datetime
import time
import xarray as xr
from convlstm.src.utils.one_hot_encoder import make_one_hot
from convlstm.src.utils.opt_flow_predictor import opt_flow_predict
from convlstm.src.utils.torch_xarray_helper_functions import merge_types_dataarray

from convlstm.src.utils.conv_lstm_predictor import convlstm_predict
from convlstm.src.models.gan_models import MDGAN_S1_G, MDGAN_S2_G


from convlstm.src.data.load_ERA_cloud_distribution import retrieve_ec_map


n_timesteps = 16
batch_size = 4

path_results = '/media/data/CloudCast_Results/'

path = "/media/oldL/data/"
test_set = CloudDataset(nc_filename=path + 'data_sets/test/ct_and_sunhours_test.nc',  # '/2018M1_CT.nc'  # '/TestCloudHourly.nc'
                        root_dir=path,
                        sequence_start_mode='unique',
                        n_lags=n_timesteps,
                        transform=pre_process_data,
                        normalize=False,
                        nan_to_zero=True,
                        run_tests=False,
                        return_timestamp=True,
                        restrict_classes=True,
                        day_only=False,  # Tr
                        include_metachannels=True,
                        frames=n_timesteps * 2)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=batch_size,
    num_workers=1,
    shuffle=False
)


# initialize parameters and dictionaries
NUM_CLASSES = 4
hourly = True  # save predictions as resampled hourly means or raw 15min intervals
hourly_opt = False  # use hourly input for opt flow (does not work very well)
time_now = str(datetime.datetime.now())[0:16].replace(' ', '_')
multiprocess = False
NUM_CHANNELS_IN = 5  # 5
softmax=True

model_dict = {
    'Persistence': [],
    'ECMWF': [],
    'OptFlow': [],
    'ConvLSTM': []
}
predictions_dict = {'Persistence': {},
                    'ECMWF': {},
                    'OptFlow': {},
                    'ConvLSTM': {},
                    'Target': {}}

# CONVLSTM AutoENcoder
device = torch.device("cuda")

from convlstm.src.convLSTM_lightning import ConvLSTMModel, Trainer
model = ConvLSTMModel()
trainer = Trainer(gpus=2, distributed_backend='dp', early_stop_callback=False,
                  resume_from_checkpoint='/home/local/DAC/ahn/Documents/dcwis.convlstm/model_checkpoints/model.ckpt')
model.freeze()

# GAN STAGE 1 AND 2
# filename = path +'/ECMWF_Data/operational/final/' + '/ECMWF_Full_Stacked.nc'  # clouds_ERA_01_05
# df_final = xr.open_dataset(filename)


gan_stage_1_model = netG_S1 = MDGAN_S1_G(32).to(device)
gan_stage_1_model.load_state_dict(
    torch.load(path + "/timelapse_gan/generator_s1_state_dict_best_softmax_tuned.pth")) # convLSTM_future_3D_CE_state_dict # convLSTM_CE_unet

gan_stage_2_model = netG_S2 = MDGAN_S2_G(32).to(device)
gan_stage_2_model.load_state_dict(
    torch.load(path + "/timelapse_gan/generator_s2_state_dict_best_softmax_tuned.pth")) # convLSTM_future_3D_CE_state_dict # convLSTM_CE_unet
" REMEMBER TO CHANGE NAME HERE ^ "


opt_flow_params = {
    "tau": 0.3,
    "lambda": 0.21,
    "theta": 0.5,
    "n_scales": 3,
    "warps": 5,
    "epsilon": 0.01,
    "innnerIterations": 10,
    "outerIterations": 2,
    "scaleStep": 0.5,
    "gamma": 0.1,
    "medianFiltering": 5
}
#  " OPTIMAL OPT FLOW PARAMETERS "


" LOAD ECMWF "
include_ECMWF = False
if include_ECMWF:
    filename = path +'data_sets/ECMWF_Data/operational/final/' + '/ECMWF_Full_Stacked.nc'  # clouds_ERA_01_05
    df_final = xr.open_dataset(filename).load()

import pandas as pd

def main(idx_start):
    if hourly==False:
        assert (hourly == hourly_opt)

    print('Initializing individual run')
    start = time.time()
    chunk_counter = idx_start
    ind_start = time.time()
    for i, (imgs_X, imgs_Y, time_stamp) in enumerate(test_loader):
        if i > idx_start and i < len(test_loader)-1: # to allow for multiprocessing
            print('Iteration %i out of %i ' % (i, len(test_loader)))

            # timestamps
            ts = time_stamp.numpy()  # .astype('int')
            ts = ts.astype('datetime64[s]')
            target_times = ts[:, -16:]
            #print(target_times)

            # if np.datetime64("2018-09-02T03:00:00") in target_times:

            # from convlstm.src.utils.visualize import Visualizer

            " GAN PREDICTION "
            with torch.no_grad():

                batch = 0
                # vis = Visualizer()

                one_hot_input = make_one_hot(imgs_X[:, 0, :, :, :].unsqueeze(1).cuda().squeeze().long(), 4)

                # input_video_gan = 2 * (imgs_X[:, 0, :, :, :].unsqueeze(1).cuda() / 3) - 1
                # vis.create_timeseries_plot(input_video_gan[0, 0, :, :, :].cpu().numpy(), name='input_norm', frames=16)

                # Stage 1
                out_gan1 = gan_stage_1_model(one_hot_input)
                # class_pred_gan1 = (out_gan1 * -1.5 + 1.5).round()  # renormalize and round to nearest integer
                # Stage 2
                class_pred_gan1 = torch.argmax(out_gan1, 1)
                class_pred_one_gan1 = make_one_hot(class_pred_gan1.squeeze().long(), NUM_CLASSES)

                one_hot_fake_g1 = make_one_hot(class_pred_gan1, 4)

                out_gan2 = gan_stage_2_model(one_hot_fake_g1)

                # class_pred_gan = (out_gan2 * -1.5 + 1.5).round()  # renormalize and round to nearest integer

                class_pred_gan = torch.argmax(out_gan2, 1)
                class_pred_one_gan = make_one_hot(class_pred_gan.squeeze().long(), NUM_CLASSES)

                # vis.create_timeseries_plot(class_pred_gan1[batch, :, :, :].cpu().numpy(), name='gan_stage1', frames=16)
                # vis.create_timeseries_plot(class_pred_gan[batch, :, :, :].cpu().numpy(), name='gan_stage2', frames=16)
                # vis.create_timeseries_plot(imgs_X[batch, 0, :, :, :].cpu().numpy(), name='input', frames=16)
                # vis.create_timeseries_plot(imgs_Y[batch, 0, :, :, :].cpu().numpy(), name='output', frames=16)

            # X and Y Variables
            imgs_X = Variable(imgs_X.cuda()) #.cuda())  # .half()) # .half()
            imgs_X = imgs_X.permute(0, 2, 1, 3, 4)
            #target = Variable(imgs_Y).cuda()  # .half()
            target = Variable(imgs_Y[:, 0, :, :, :]).cuda()  # .hal
            target_one = make_one_hot(target.long(), NUM_CLASSES)

            # DL predictiong

            with torch.no_grad():
                if NUM_CHANNELS_IN > 4:
                    # out = convlstm_predict(model, imgs_X, metachannels=imgs_Y[:, 1, :, :, :], softmax=softmax)
                    out = model.forward(imgs_X[:, :, 0, :, :])
                else:
                    out = model.forward(imgs_X[:, :, 0, :, :])

                out = out[:, :, -16:, :, :]
                    # out = convlstm_predict(model, imgs_X, softmax=softmax)

            # if hourly:
            #     out = target_times

            class_pred = torch.argmax(out, 1)
            class_pred_one = make_one_hot(class_pred.long(), NUM_CLASSES)

            " PREPARE ECMWF DATA AND MAKE OPTFLOW PREDICTION "
            if include_ECMWF:
                ecmwf_matrix = np.zeros((target_one.shape))
            # print(ecmwf_matrix.shape)

            # if hourly:
            #     opt_flow_tensor = torch.zeros(size=imgs_X[:, -4:, 0, :, :].shape, dtype=torch.int)
            # else:
            opt_flow_tensor = torch.zeros(size=imgs_X[:, :, 0, :, :].shape, dtype=torch.int)

            # loop over every batch
            try:
                if include_ECMWF:
                    for k in range(batch_size):
                        hourly_time_stamp = target_times[k, 3::4]
                        exact_times_ec = False
                        " exact timeslots for ECMWF "
                        if exact_times_ec:
                            l = pd.to_datetime(hourly_time_stamp+ 15 * 60)
                            l_idx = [int(x) for x in l.hour]
                            if l_idx == [1, 2, 3, 4] or l_idx == [13, 14, 15, 16]:
                                print('Calculating ECMWF forecast')

                                for idx, l in enumerate(hourly_time_stamp):
                                    subset = df_final.sel(ValueDate=l, method='nearest')
                                    print(l)
                                    ecmwf_matrix[k, idx, :, :] = retrieve_ec_map(subset, binary=False, plot=False, type='all', threshold=0.1)
                        else:
                            for idx, l in enumerate(hourly_time_stamp):
                                subset = df_final.sel(ValueDate=l, method='nearest')
                                ec_cover = retrieve_ec_map(subset, binary=False, plot=False, type='all')
                                ecmwf_matrix[k, :, idx, :, :] = ec_cover


                            """ TESTINNG """
                            #
                            # vals = ec_cover[2, :, :]
                            # # vals = ec_cover[:, :, 2]
                            # vals[vals == 0] = np.nan
                            # crs = area_def.to_cartopy_crs()
                            #
                            # # plt.imshow(vals)
                            #
                            # plt.figure(figsize=(8, 6))
                            # ax = plt.axes(projection=crs)
                            #
                            # ax.background_img(name='BM',
                            #                   resolution='low')  # BM is a custom image collected from https://neo.sci.gsfc.nasa.gov/view.php?datasetId=BlueMarbleNG-TB
                            #
                            # plt.imshow(vals, cmap='Blues', transform=crs, extent=crs.bounds, origin='upper')

                            """ TESTINNG END """

                    if hourly_opt: # here we use hourly series as input instead of 15-min as we are predicting 15minute intervals
                        opt_flow_prediction = opt_flow_predict(imgs_X[k, -8::4, 0, :, :].detach().cpu(),
                                                                       'tvl1',
                                                                       save_flow=False, params=dict(opt_flow_params),
                                                                       future=4) \
                            .round()

                        for f in range(4):
                            opt_flow_tensor[k, f:(f+4), :, :] = opt_flow_prediction[f].unsqueeze(0).repeat(4, 1, 1)


                    else:
                        opt_flow_tensor[k, :, :, :] = opt_flow_predict(imgs_X[k, :, 0, :, :].detach().cpu(),
                                                                       'tvl1',
                                                                       save_flow=False, params=dict(opt_flow_params),
                                                                       future=16) \
                            .round()

                    # vis.create_timeseries_plot(opt_flow_tensor[k, :, :, :].cpu().numpy(), name=str(k) + '_OptFlow',
                    #                            frames=16)
                    #
                    # vis.create_timeseries_plot(imgs_X[k, :, 0, :, :].detach().cpu().numpy(), name=str(k) + '_Input',
                    #                            frames=16)


            except KeyError as e:
                print('Could not retrieve ECMWF data due to %s ' % e)
                continue
            except IndexError as e:
                print('Could not retrieve ECMWF data due to %s ' % e)
                continue

            opt_flow_one_hot = make_one_hot(opt_flow_tensor.long().cuda(), NUM_CLASSES)

            if include_ECMWF:
                var = np.argmax(ecmwf_matrix, 1)
                var = var.astype('float')
                ec_category = torch.tensor(var).cuda()

                ec_tensor = torch.tensor(ecmwf_matrix).cuda()

                ec_one_hot = make_one_hot(ec_category.long(), NUM_CLASSES)

                ecmwf_dict = {'one_hot': ec_one_hot.detach().cpu(),
                              'probabilities': ec_tensor.detach().cpu(),  # consider one-hot for probabilities!
                              'categorical': ec_category.detach().cpu()}

            benchmark_pred = imgs_X[:, -1, 0, :, :].unsqueeze(1)
            benchmark_pred = benchmark_pred.repeat(1, n_timesteps, 1, 1) # NOTE! #  // 4
            benchmark_pred_one = make_one_hot(benchmark_pred.long(), NUM_CLASSES)


            benchmark_dict = {'one_hot': benchmark_pred_one.detach().cpu(),
                              'probabilities': benchmark_pred_one.detach().cpu(),
                              'categorical': benchmark_pred.detach().cpu()}

            opt_flow_dict = {'one_hot': opt_flow_one_hot.detach().cpu(),
                             'probabilities': opt_flow_one_hot.detach().cpu(),
                             'categorical': opt_flow_tensor.detach().cpu()}

            # torch.softmax(out_gan2, 1)
            gan_dict = {'one_hot': class_pred_one_gan.detach().cpu(),
                        'probabilities': class_pred_one_gan.detach().cpu(),
                        'categorical': class_pred_gan.squeeze().detach().cpu()}

            gan_stage1_dict = {'one_hot': class_pred_one_gan1.detach().cpu(),
                        'probabilities': class_pred_one_gan1.detach().cpu(),
                        'categorical': class_pred_gan1.squeeze().detach().cpu()}


            convlstm_dict = {'one_hot': class_pred_one.detach().cpu(),
                             'probabilities': out.detach().cpu(),
                             'categorical': class_pred.detach().cpu()}

            target_dict = {'one_hot': target_one.detach().cpu(),
                           'probabilities': target_one.detach().cpu(),
                           'categorical': target.detach().cpu()}

            predictions_dict = {
                'Persistence': benchmark_dict,
                # 'ECMWF': ecmwf_dict,
                'OptFlow': opt_flow_dict,
                'GAN_Stage1': gan_stage1_dict,
                'GAN_Stage2': gan_dict,
                'ConvLSTM': convlstm_dict,
                'Target': target_dict
            }

            " PREDICTIONS SAVING TO XARRAY"
            predictions_xarray_dict = {
                'Persistence': {},
                # 'ECMWF': {},
                'OptFlow': {},
                'GAN': {},
                'GAN_Stage1': {},
                'GAN_Stage2': {},
                'ConvLSTM': {},
                'Target': {}
            }
            for mdl in predictions_dict.keys():
                #print(mdl)
                predictions_dict[mdl]['categorical'] = predictions_dict[mdl]['categorical'].unsqueeze(1) \
                    .repeat(1, 4, 1, 1, 1)

                all_types_xr = merge_types_dataarray(predictions_dict[mdl], batch_size, target_times)
                #predictions_dict[mdl].shape

                predictions_xarray_dict[mdl] = all_types_xr.copy()

            # concatenate into dataset
            temp_xr_df = xr.Dataset(data_vars={
                'Persistence': predictions_xarray_dict['Persistence'],
                # 'ECMWF': predictions_xarray_dict['ECMWF'],
                'OptFlow': predictions_xarray_dict['OptFlow'],
                'GAN_Stage1': predictions_xarray_dict['GAN_Stage1'],
                'GAN_Stage2': predictions_xarray_dict['GAN_Stage2'],
                'ConvLSTM': predictions_xarray_dict['ConvLSTM'],
                'Target': predictions_xarray_dict['Target']
            },
                coords=all_types_xr.coords)

            if hourly:
                #TODO: MAKE SURE THIS WORKS!
                # temp_xr_df['time'] += np.timedelta64(15, 'm')  # we do temporarily to align one-, two-, three- and four hour
                #                                                # ahead forecasts
                # unique_hours, hours_idx = np.unique(temp_xr_df.time.dt.hour, return_index=True)
                # hours_to_subset = temp_xr_df.time[hours_idx[1:]]
                # temp_xr_df = temp_xr_df.sel(time=hours_to_subset)
                #
                # temp_xr_df['time'] -= np.timedelta64(15, 'm')
                #
                # temp_xr_df = temp_xr_df.sel(time=hours_to_subset)
                final_target_times = np.array([x[3::4] for x in target_times]).ravel()
                temp_xr_df = temp_xr_df.sel(time=final_target_times)

            if i < 1:
                full_xr_df = temp_xr_df.copy()

            # elif i == 8:
            #     print('ll')t
            #     import pandas as pd
            #     df_v = pd.Series(full_xr_df.time)

            elif i % 50 == 0:
                print("50 iterations took =", (time.time() - ind_start))

                if hourly:
                    paths = path_results + '/predictions/all_models_' + time_now + '_chunk_' + str(chunk_counter+1) + '_hourly.nc'
                else:
                    paths = path_results + '/predictions/all_models_' + time_now + '_chunk_' + str(chunk_counter+1) + '.nc'
                #full_xr_df = full_xr_df.chunk(chunks=30)
                full_xr_df = full_xr_df.sortby('time')
                full_xr_df.to_netcdf(paths)

                chunk_counter += 1

                del full_xr_df
                full_xr_df = temp_xr_df.copy()
                ind_start = time.time()

            else:

                full_xr_df = xr.concat([full_xr_df, temp_xr_df], dim='time')


    # final predictions

    if hourly:
        paths = path_results + '/predictions/all_models_' + time_now + '_chunk_' + str(chunk_counter + 1) + '_hourly.nc'
    else:
        paths = path_results + '/predictions/all_models_' + time_now + '_chunk_' + str(chunk_counter + 1) + '.nc'

    full_xr_df = full_xr_df.sortby('time')
    full_xr_df.to_netcdf(paths)

    print("Individual run took =", (time.time() - start))

    # todo: remember when loading the dataarrays to exclude the additional repeated channels from tensor and categorical


# from multiprocessing import Pool

if __name__ == '__main__':
    start = time.time()

    processes = 2
    iterations = len(test_loader)

    iterable_range = iterations // 4
    iterable_start = [x*iterable_range for x in range(processes)]

    # if multiprocess == True:
    #     with Pool(processes=4) as pool:
    #         pool.map(main, iterable_start)
    #     # pool.starmap(main, iterable_start)
    #     print("Entire parallel run took =", (time.time() - start))
    #
    # else:
    main(-1)
    print("Entire non-parallel run took =", (time.time() - start))

