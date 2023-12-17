import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)


        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes


    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_center = 2*band.view(-1,1)
        band_pass_right= torch.flip(band_pass_left,dims=[1])


        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)


        band_pass = band_pass / (2*band[:,None])


        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, filters, is_first=False):
        super(ResBlock, self).__init__()
        
        if in_channels != filters:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=1
            )
            
        else:
            self.downsample = False
        
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filters,
            kernel_size=3,
            padding=1
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=3,
            padding=1
        )
        
        if is_first:
            self.model = nn.Sequential(
                self.conv1,
                nn.BatchNorm1d(num_features=filters),
                nn.LeakyReLU(),         
                self.conv2,
            )            
        else:
            self.model = nn.Sequential(
                nn.BatchNorm1d(num_features=in_channels),
                nn.LeakyReLU(),
                self.conv1,
                nn.BatchNorm1d(num_features=filters),
                nn.LeakyReLU(),         
                self.conv2,
            )
        
        self.max_pool = nn.MaxPool1d(kernel_size=3)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.fms = nn.Sequential(
            nn.Linear(in_features=filters, out_features=filters),
            nn.Sigmoid()
        )
        
        
    def forward(self, x):
        identity = x
        x = self.model(x)
        if self.downsample:
            identity = self.conv_downsample(identity)
        x = self.max_pool(x + identity)
        
        f = self.avg_pool(x).squeeze(-1)
        f = self.fms(f).unsqueeze(-1)

        x = x * f + f
        return x


class RawNet2(nn.Module):
    def __init__(self, args):
        super(RawNet2, self).__init__()

        self.sinc_filter = SincConv_fast(
                out_channels=args['filters'][0],
                kernel_size=args['sinc_filter_len'],
                in_channels=args['in_channels'],
                min_low_hz=0,
                min_band_hz=0
            )
        self.after_sinc_filter = nn.Sequential(
            nn.MaxPool1d(kernel_size=3),
            nn.BatchNorm1d(num_features=args['filters'][0]),
            nn.LeakyReLU(),
        )

        self.first_res_block_stack = nn.Sequential(
                *[ResBlock(
                    in_channels=args['filters'][0] if i == 0 else args['filters'][1],
                    filters=args['filters'][1],
                    is_first=True if i == 0 else False,
                ) for i in range(args['res_blocks_num'][0])]
            )
        
        self.second_res_block_stack = nn.Sequential(
                *[ResBlock(
                    in_channels=args['filters'][1] if i == 0 else args['filters'][2],
                    filters=args['filters'][2]
                ) for i in range(args['res_blocks_num'][1])]
            )
        
        self.before_gru = nn.Sequential(
            nn.BatchNorm1d(num_features=args['filters'][2]),
            nn.LeakyReLU()
        )
        self.gru = nn.GRU(
            input_size=args['filters'][2],
            hidden_size=args['gru_hidden_size'],
            num_layers=args['gru_num_layers'],
            batch_first=True
        )
        
        self.fc_output1 = nn.Linear(
            in_features=args['gru_hidden_size'],
            out_features=1024
        )
        
        self.fc_output2 = nn.Linear(
            in_features=1024,
            out_features=2
        )
    
    def forward(self, x):
        x = self.sinc_filter(x)
        # x = self.after_sinc_filter(torch.abs(x))
        x = self.after_sinc_filter(x)


        x = self.first_res_block_stack(x)
        x = self.second_res_block_stack(x)
        
        x = self.before_gru(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        
        x = x[:,-1,:]
        
        x = self.fc_output1(x)
        x = self.fc_output2(x)        
        return x