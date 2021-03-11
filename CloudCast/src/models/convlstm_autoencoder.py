import torch
import torch.nn as nn
from torch.autograd import Variable

from CloudCast.src.utils.one_hot_encoder import make_one_hot
import torch.nn.functional as F

# from convlstm.src.utils.LayerNorm import LayerNorm

class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        zeros = Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width))
        return (Variable(torch.nn.init.xavier_uniform_(zeros)).cuda(),#.half(),
                Variable(torch.nn.init.xavier_uniform_(zeros)).cuda())#.half())
        # TRY TO SEE WHAT THIS RETURNS

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2).cuda()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)) \
            .cuda()

    def forward(self, x1, x2):

        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class DeepAutoencoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan, input_size=(16, 16), n_classes=4):

        super(DeepAutoencoderConvLSTM, self).__init__()

        self.convlstm_initial = ConvLSTMCell(input_size=input_size,
                                             input_dim=nf * 4,
                                             hidden_dim=nf * 4,
                                             kernel_size=(3, 3),
                                             bias=True)

        self.convlstm_intermediary = ConvLSTMCell(input_size=input_size,
                                                  input_dim=nf * 4,
                                                  hidden_dim=nf * 4,
                                                  kernel_size=(3, 3),
                                                  bias=True)

        self.convlstm_final = ConvLSTMCell(input_size=input_size,
                                           input_dim=nf * 4,
                                           hidden_dim=nf * 4,
                                           kernel_size=(3, 3),
                                           bias=True)

        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=in_chan,     # could set this five
                      out_channels=nf,  # 11
                      bias=False,
                      kernel_size=(3, 3),
                      padding=(1, 1),
                      stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(nf)
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=nf,  # could set this to 4 and then use one-hot encoded inputs!
                      out_channels=nf*2,  # 11
                      bias=False,
                      kernel_size=(3, 3),
                      padding=(1, 1),
                      stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(nf*2)
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=nf*2,  # could set this to 4 and then use one-hot encoded inputs!
                      out_channels=nf*4,  # 11
                      bias=False,
                      kernel_size=(3, 3),
                      padding=(1, 1),
                      stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(nf*4)
        )
        self.downsampler = nn.MaxPool2d(2)
        #           # try convolution with stride 2


        self.decoder1 = up(512, 128, bilinear=False)
        self.decoder2 = up(256, 64, bilinear=False)
        self.decoder3 = up(128, 4, bilinear=False)

        self.layernorm1 = torch.nn.LayerNorm([64, 128, 128]).cuda()
        self.layernorm2 = torch.nn.LayerNorm([128, 64, 64]).cuda()
        self.layernorm3 = torch.nn.LayerNorm([256, 32, 32]).cuda()


    def autoencoder(self, x, seq_len, h_t, c_t, h_t2, c_t2, h_t3, c_t3, future=False, metachannels=None):
        outputs = []

        if future:
            output = x

        for t in range(seq_len):
            if future:
                output = torch.softmax(output, dim=1)
                output = torch.argmax(output, dim=1) \
                    .unsqueeze(1)
                output = make_one_hot(output.long(), 4) \
                    .squeeze(2)  # THIS DOES NOT WORK WITH METACHANNEL!

                if metachannels is not None:
                    output = torch.cat([output, metachannels[:, t, :, :].unsqueeze(1).cuda()], 1)

                x1_c = self.encoder1(output)
                # x1_c = self.layernorm1(x1_c)

            else:
                x1_c = self.encoder1(x[:, t, :, :, :])  # 4, 16, 5, 128, 128  == correct?
                # x1_c = self.layernorm1(x1_c)

            x1 = self.downsampler(x1_c)

            x2_c = self.encoder2(x1)
            # x2_c = self.layernorm2(x2_c)
            x2 = self.downsampler(x2_c)

            x3_c = self.encoder3(x2)
            # x3_c = self.layernorm3(x3_c)
            x3 = self.downsampler(x3_c)

            h_t, c_t = self.convlstm_initial(input_tensor=x3,
                                             cur_state=[h_t, c_t])  # we could concat to provide skip conn here
            h_t2, c_t2 = self.convlstm_intermediary(input_tensor=h_t,
                                                    cur_state=[h_t2, c_t2])
            h_t3, c_t3 = self.convlstm_final(input_tensor=h_t2,
                                             cur_state=[h_t3, c_t3])
            # print(h_t3.shape)
            output = self.decoder1(h_t3, x3_c)
            # print(output.shape)
            output = self.decoder2(output, x2_c)
            # print(output.shape)
            output = self.decoder3(output, x1_c)  # maybe 4 to 64 is a bit rough
            # print(output.shape)

            outputs += [output]

        return outputs, output

    def forward(self, x, future_seq=0, metachannels=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (t, b, c, h, w)
        """
        # initialize hidden state
        # TODO: Try similar convolutions between encoder and decoder
        h_t, c_t = self.convlstm_initial.init_hidden(x.size(0))  # maybe do not initialize to zero but do glorot
        h_t2, c_t2 = self.convlstm_intermediary.init_hidden(x.size(0))
        h_t3, c_t3 = self.convlstm_final.init_hidden(x.size(0))

        seq_len = x.size(1)

        # METACHANNEL
        if x.shape[2] > 1:
            x_onehot = make_one_hot(x[:, :, 0, :, :].squeeze(2).long(), 4) \
                .permute(0, 2, 1, 3, 4)
            x = torch.cat([x_onehot, x[:, :, 1:, :, :]], 2)  # MAKE SURE THIS WORKS
        else:
            x = make_one_hot(x.squeeze(2).long(), 4)  # DO THIS FOR MSE!
            x = x.permute(0, 2, 1, 3, 4)

        # input sequence length
        # torch.Size([4, 16, 5, 128, 128])
        outputs, output = self.autoencoder(x, seq_len, h_t, c_t, h_t2, c_t2, h_t3, c_t)
        outputs = torch.stack(outputs, 1)

        # Future predictions based on transformed input
        if future_seq != 0:
            outputs_future, _ = self.autoencoder(output, future_seq, h_t, c_t, h_t2, c_t2, h_t3, c_t, future=True,
                                                 metachannels=metachannels)
            outputs_future = torch.stack(outputs_future, 1)
            outputs = torch.cat([outputs, outputs_future], 1)

        outputs = outputs.permute(0, 2, 1, 3, 4)  # .squeeze(2)            # torch.Size([2, 128, 16, 128, 128])

        return outputs
