
"""
Model Fitting module
"""
# Author: Andreas Holm <ahn@eng.au.dk>

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import Variable


from convlstm.src.utils.lookahead_pytorch import Lookahead
from convlstm.src.utils.radam import RAdam

from dcwis.satelliteloader.DataLoader import CloudDataset

from dcwis.satelliteloader.utils.pre_process import pre_process_data

# from utils.loss_functions import custom_cross_entropy

from convlstm.src.utils.one_hot_encoder import make_one_hot
from convlstm.src.utils.loss_functions import gdl2


from convlstm.src.models.DeepAE_ConvLSTM import DeepAutoencoderConvLSTM
import numpy as np
from tensorboardX import SummaryWriter

# integration to evaluator project for quick evaluation (insert into tensorboard)

import matplotlib.pyplot as plt
import matplotlib.lines as ll


# def plot_grad_flow(named_parameters):
#     ave_grads = []
#     layers = []
#     for n, p in named_parameters:
#         if(p.requires_grad) and ("bias" not in n):
#             layers.append(n)
#             ave_grads.append(p.grad.abs().mean())
#     plt.plot(ave_grads, alpha=0.3, color="b")
#     plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
#     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
#     plt.xlim(xmin=0, xmax=len(ave_grads))
#     plt.yscale('log')
#     plt.ylim(bottom=1e-9, top=1e-1)  # Keep scale fixed
#     plt.xlabel("Layers")
#     plt.ylabel("average gradient")
#     plt.title("Gradient flow")
#     plt.grid(True)
#     plt.show()
    # plt.savefig("/data/grad_flow/grad_flow.png")



#
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.4, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.4, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    #plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.yscale('log')
    plt.ylim(bottom=1e-9, top=1e-1)  # Keep scale fixed
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.legend([ll.Line2D([0], [0], color="c", lw=4),
                ll.Line2D([0], [0], color="b", lw=4),
                ll.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
    #plt.savefig("/data/grad_flow/grad_flow2.png")


def run_model_training(save_model, save_predictions, tensorboard_on, gradient_flow_checking,
                       n_epochs):
    #path = "/data1"
    # path = "/data"
    path = "/media/oldL/data"

    if tensorboard_on:
        writer = SummaryWriter(path + '/logs/')
    # path = 'D:/final_ct_dataset/'

    if save_predictions:
        from convlstm.src.visualization.save_performance_plot import save_performance_plot_ConvLSTM
        from convlstm.src.visualization.save_images_cartopy import save_images_cartopy

    NUM_CLASSES = 4
    BATCH_SIZE = 4
    NORMALIZE = False
    NUM_CHANNELS_IN = 5  # 5

    train_loss_lst = []
    test_loss_lst = []
    sample_interval = 500 // BATCH_SIZE * 2  # 500 // BATCH_SIZE * 2
    batches_done = 0
    initial_runs = 100 // BATCH_SIZE * 2

    # Initialize data loader
    train_set = CloudDataset(nc_filename=path + '/data_sets/train/ct_and_sunhours_train.nc',  # TrainCloud
                             root_dir=path + '/data_sets',
                             sequence_start_mode='unique',
                             n_lags=16,
                             transform=pre_process_data,
                             normalize=NORMALIZE,
                             nan_to_zero=True,
                             restrict_classes=True,
                             return_timestamp=False,
                             run_tests=False,
                             day_only=False,  # Tr
                             include_metachannels=True,
                             frames=32)
    # find out how much is loaded into memory!!
    # february looks better than Jan! Loss works better

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        # num_workers=8,
        shuffle=True)

    # model params
    device = torch.device("cuda")
    model = DeepAutoencoderConvLSTM(nf=64, in_chan=NUM_CHANNELS_IN, n_classes=NUM_CLASSES).to(device)

    # TRY THIS!
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


    loss_m = 'gdl'

    if loss_m == 'L1':
        criterion = torch.nn.L1Loss().cuda()
        base_lr = 1e-7
    elif loss_m == 'weighted_ce':
        #base_lr = 3e-4
        base_lr = 2e-4
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1 / 0.3, 1 / 0.3, 1 / 0.1, 1 / 0.3]), reduction='mean').cuda()  # .half() # consider using MSE instead!!
    elif loss_m == 'gdl':
        criterion = gdl2
    # weights are based on distributional dispersion of data

    # SCHEDULER
    #base_lr = 0.5e-3

    # optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))  # 0.5, 0.9
    #optimizer_adam = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))  # 0.5, 0.9
    opt_adam = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.98), eps=1e-9)
    #optimizer_radam = RAdam(model.parameters(), lr=base_lr)
    # optimizer = Lookahead(optimizer_adam)

    from convlstm.src.utils.NoamOpt import NoamOpt

    optimizer = NoamOpt(model.parameters(), 1, 400, opt_adam) # try

    # criterion = torch.nn.CrossEntropyLoss().cuda()
    # criterion = torch.nn.MSELoss().cuda() # consider using MSE instead!!

    lambda_pixel = 1

    # try this one : SmoothL1Loss
    # calculate these weights based on historical statistics!!

    train_every_batch = False
    scheduler_on = False
    pred_only_visu = True

    # TRY THIS!!!
    # def lr_scheduler(optimizer, epoch):
    #     if epoch < 7:
    #         return optimizer
    #     #elif epoch < n_epochs // 2:
    #     elif epoch < 15:
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = 0.0001
    #     elif epoch < 35:
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = 0.00005
    #     else:
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = 0.00001
    #     return optimizer

    import time
    for epoch in range(n_epochs):
        #optimizer = lr_scheduler(optimizer, epoch)
        for i, (imgs_X, imgs_Y) in enumerate(train_loader):
            #TODO: Tensorboard - add timings for training and also for writing images

            # run_lr = scheduler.get_lr()
            #input = Variable(imgs_X.unsqueeze(2).cuda())  # .half()) # .half()

            #doing future predictions?

            if imgs_X.shape[1] > 1 and len(imgs_X.shape) > 4:
                # Prepare input data
                input = Variable(imgs_X.cuda())  # .half()) # .half()
                output_train = input[:, 0, 1:, :, :]  # why [:, 1??
                # 1: to ensure we are looking at final 15 timesteps of input
                output_train = output_train.long()
                target = Variable(imgs_Y[:, 0, :, :, :]).cuda()  # .hal
                input = input.permute(0, 2, 1, 3, 4)
                #torch.Size([4, 2, 16, 128, 128])

            else:
                input = Variable(imgs_X.unsqueeze(2).cuda())  # .half()) # .half()
                output_train = input[:, 1:, :, :, :]  # why [:, 1??
                output_train = output_train.long().squeeze(2)
                # ss
                target = Variable(imgs_Y).cuda()  # .half()

                #torch.Size([4, 16, 128, 128])

                #torch.Size([4, 16, 1, 128, 128])



            # target = make_one_hot(target.long(), NUM_CLASSES) # DO THIS FOR MSE!
            # clear previous gradients
            # optimizer.zero_grad()
            optimizer.optimizer.zero_grad()
            # Future prediction
            if batches_done > initial_runs:  # we could also optimize both separately? So half weight on training, half on testing
                # multiple steps-ahead sequence prediction
                if NUM_CHANNELS_IN > 4:
                    out = model(input, future_seq=15, metachannels=imgs_Y[:, 1, :, :, :]) # only 15 because final prediction from input is one-step ahead, then remaining 15 steps is needed
                else:
                    out = model(input, future_seq=15)
                # INPUT SHOULD ALSO BE  imgs_Y.SUNHOURS
                #loss_train = criterion(out[:, :, :15, :, :], output_train.long())


                if loss_m == 'L1':
                    output_test_C = make_one_hot(target.long())
                    out = torch.softmax(out, dim=1)
                    loss_test = criterion(out[:, :, -16:, :, :], output_test_C)
                elif loss_m == 'weighted_ce':
                    loss_test = criterion(out[:, :, -16:, :, :].float(), target.long())
                elif loss_m == 'gdl':
                    output_test_C = make_one_hot(target.long())
                    out = torch.softmax(out, dim=1)
                    loss_test = criterion(out[:, :, -16:], output_test_C, alpha=1)

                loss = loss_test# + loss_train

                loss.backward()

                if gradient_flow_checking:
                    print('lets print gradient flow')
                    plot_grad_flow(model.named_parameters())

                test_loss = loss_test.item()
                if tensorboard_on:
                    writer.add_scalar('16-Step/Loss', test_loss, batches_done)

            else:
                # one-step ahead prediction training, used for initial weights updating
                out = model(input)  # .permute(0, 2, 1, 3, 4)

                # MAYBE NOT PIXEL GROUND TRUTHS BUT THE SOFTMAX PROBABILITIES

                # gdl = loss(, output_train)
                # gdl.backward()

                if loss_m == 'L1':
                    output_train_C = make_one_hot(output_train)
                    out = torch.softmax(out, dim=1)
                    loss_train = criterion(out[:, :, 0:15, :, :], output_train_C)
                elif loss_m == 'weighted_ce':
                    loss_train = criterion(out[:, :, 0:15, :, :].float(), output_train.long())  # should target be one-hot?
                elif loss_m == 'gdl':
                    output_train_C = make_one_hot(output_train)
                    out = torch.softmax(out, dim=1)
                    loss_train = criterion(out[:, :, 0:15], output_train_C, alpha=1)

                loss_train.backward()
                test_loss = np.nan

                if gradient_flow_checking:
                    print('lets print gradient flow')
                    plot_grad_flow(model.named_parameters())

            print(
                "[Epoch %d/%d] [Batch %d/%d] [One-Step loss: %f] [16-Step loss: %f]"
                % (epoch + 1, n_epochs, i, len(train_loader), loss_train.item(), test_loss)
            )

            if tensorboard_on:
                writer.add_scalar('One-Step/Loss', loss_train.item(), batches_done)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], batches_done)

            train_loss_lst.append([batches_done, loss_train.item()])
            test_loss_lst.append([batches_done, test_loss])
            # update using calculated gradients
            # optimizer.step()
            optimizer.optimizer.step()
            # print('Learning rate is %s compared to %s last iteration' % (scheduler.get_lr(), run_lr))

            batches_done = epoch * len(train_loader) + i

            if batches_done % sample_interval == 0 and batches_done > initial_runs and save_predictions==True:  # save output images every sampling interval
                if NUM_CHANNELS_IN > 4:
                    out = model(input, future_seq=15, metachannels=imgs_Y[:, 1, :, :, :])
                else:
                    out = model(input, future_seq=15)

                out = nn.Softmax(dim=1).cuda()(out).detach().cpu()
                out = torch.argmax(out, 1).unsqueeze(1)

                tensor_input = imgs_X.detach().cpu().unsqueeze(1)
                tensor_image_GT = imgs_Y.detach().cpu().unsqueeze(1)

                # try to make this part faster!
                # with torch.no_grad():
                #     for name, param in model.named_parameters():
                #         writer.add_histogram(name, param.clone().cpu().data.numpy(), batches_done)

                save_images_cartopy(tensor_image_output=out,
                                    tensor_image_input=tensor_input,
                                    tensor_image_ground_truth=tensor_image_GT,
                                    pred_only=pred_only_visu,
                                    high_res_map=False,  # true
                                    lines=False,  # true
                                    epoch=epoch,
                                    batch=i,
                                    path=path + '/saved_preds/')
                if save_model:
                    torch.save(model.state_dict(), path + '/models_saved/deep_autoencoder_convLSTM_sunhours_gdl.pth')

    train_loss_all = np.array(train_loss_lst)
    test_loss_all = np.array(test_loss_lst)

    if save_predictions:
        save_performance_plot_ConvLSTM(train_loss_all, test_loss_all)

    if save_model:
        torch.save(model.state_dict(), path + '/models_saved/deep_autoencoder_convLSTM_sunhours_gdl.pth')


if __name__ == '__main__':
    run_model_training(save_model=True, save_predictions=True, tensorboard_on=False,
                       gradient_flow_checking=False, n_epochs=200)

# writer.close()

# np.savetxt('/data/training_loss_np.txt', train_loss_all)
# np.savetxt('/data/tes_loss_np.txt', test_loss_all)
# ll = np.loadtxt('/data/tes_loss_np.txt')




