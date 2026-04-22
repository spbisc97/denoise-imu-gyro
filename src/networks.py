import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils import bmtm, bmtv, bmmt, bbmv
from src.lie_algebra import SO3


class BaseNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # channel dimension
        c1 = 2*c0
        c2 = 2*c1
        c3 = 2*c2
        # kernel dimension (odd number)
        k0 = ks[0]
        k1 = ks[1]
        k2 = ks[2]
        k3 = ks[3]
        # dilation dimension
        d0 = ds[0]
        d1 = ds[1]
        d2 = ds[2]
        # padding
        p0 = (k0-1) + d0*(k1-1) + d0*d1*(k2-1) + d0*d1*d2*(k3-1)
        # nets
        self.cnn = torch.nn.Sequential(
            torch.nn.ReplicationPad1d((p0, 0)), # padding at start
            torch.nn.Conv1d(in_dim, c0, k0, dilation=1),
            torch.nn.BatchNorm1d(c0, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c0, c1, k1, dilation=d0),
            torch.nn.BatchNorm1d(c1, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c1, c2, k2, dilation=d0*d1),
            torch.nn.BatchNorm1d(c2, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c2, c3, k3, dilation=d0*d1*d2),
            torch.nn.BatchNorm1d(c3, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(c3, out_dim, 1, dilation=1),
            torch.nn.ReplicationPad1d((0, 0)), # no padding at end
        )
        # for normalizing inputs
        self.register_buffer('mean_u', torch.zeros(in_dim))
        self.register_buffer('std_u', torch.ones(in_dim))
        self._zero_init_output_head()

    def forward(self, us):
        u = self.norm(us).transpose(1, 2)
        y = self.cnn(u)
        return y

    def norm(self, us):
        return (us-self.mean_u)/self.std_u

    def set_normalized_factors(self, mean_u, std_u):
        if isinstance(mean_u, int):
            mean_u = torch.tensor(mean_u, dtype=torch.float32)
        if isinstance(std_u, int):
            std_u = torch.tensor(std_u, dtype=torch.float32)
        self.mean_u.copy_(mean_u.to(self.mean_u.device))
        self.std_u.copy_(std_u.to(self.std_u.device))

    def _zero_init_output_head(self):
        # Start the residual branch at zero so the network initially behaves like
        # the static calibration path instead of injecting random corrections.
        for module in reversed(self.cnn):
            if isinstance(module, torch.nn.Conv1d):
                torch.nn.init.zeros_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                break


class GyroNet(BaseNet):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum,
        gyro_std):
        super().__init__(in_dim, out_dim, c0, dropout, ks, ds, momentum)
        self.register_buffer('gyro_std', torch.as_tensor(gyro_std, dtype=torch.float32))
        self.gyro_Rot = torch.nn.Parameter(torch.zeros(3, 3))
        self.gyro_bias = torch.nn.Parameter(torch.zeros(3))
        self.register_buffer('Id3', torch.eye(3))

    def forward(self, us):
        ys = super().forward(us)
        Rots = (self.Id3 + self.gyro_Rot).expand(us.shape[0], us.shape[1], 3, 3)
        Rot_us = bbmv(Rots, us[:, :, :3])
        return self.gyro_std*ys.transpose(1, 2) + Rot_us + self.gyro_bias.view(1, 1, 3)


class GyroNetWithoutAcc(BaseNet):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum,
        gyro_std):
        super().__init__(in_dim, out_dim, c0, dropout, ks, ds, momentum)
        self.register_buffer('gyro_std', torch.as_tensor(gyro_std, dtype=torch.float32))
        self.gyro_Rot = torch.nn.Parameter(torch.zeros(3, 3))
        self.gyro_bias = torch.nn.Parameter(torch.zeros(3))
        self.register_buffer('Id3', torch.eye(3))

    def forward(self, us):
        # set accelerometer to 0
        us[:, :, 3:6] = 0
        ys = super().forward(us)
        Rots = (self.Id3 + self.gyro_Rot).expand(us.shape[0], us.shape[1], 3, 3)
        Rot_us = bbmv(Rots, us[:, :, :3])
        return self.gyro_std*ys.transpose(1, 2) + Rot_us + self.gyro_bias.view(1, 1, 3)
    
    
class GyroNetWithRNN(BaseNet):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum,
        gyro_std):
        super().__init__(in_dim, out_dim, c0, dropout, ks, ds, momentum)
        self.register_buffer('gyro_std', torch.as_tensor(gyro_std, dtype=torch.float32))
        self.gyro_Rot = torch.nn.Parameter(torch.zeros(3, 3))
        self.gyro_bias = torch.nn.Parameter(torch.zeros(3))
        self.register_buffer('Id3', torch.eye(3))
        
        self.lstm = torch.nn.LSTM(6, 50, 3, batch_first=True, dropout=0.1)

    def forward(self, us):
        # use LSTM
        # ys = super().forward(us)
        ys = self.lstm(us)[0][:, :, :3]
        Rots = (self.Id3 + self.gyro_Rot).expand(us.shape[0], us.shape[1], 3, 3)
        Rot_us = bbmv(Rots, us[:, :, :3])
        return self.gyro_std*ys + Rot_us + self.gyro_bias.view(1, 1, 3)
    
    
class GyroNetWithCNNRNN(BaseNet):
    def __init__(self, in_dim, out_dim, c0, dropout, ks, ds, momentum,
        gyro_std):
        super().__init__(in_dim, out_dim, c0, dropout, ks, ds, momentum)
        self.register_buffer('gyro_std', torch.as_tensor(gyro_std, dtype=torch.float32))
        self.gyro_Rot = torch.nn.Parameter(torch.zeros(3, 3))
        self.gyro_bias = torch.nn.Parameter(torch.zeros(3))
        self.register_buffer('Id3', torch.eye(3))
        
        c1 = 2*c0
        c2 = 2*c1
        # c3 = 2*c2
        
        self.precnn = torch.nn.Sequential(
            torch.nn.ReplicationPad1d((0, 0)), # padding at start
            torch.nn.Conv1d(in_dim, c0, 1, dilation=1),
            torch.nn.BatchNorm1d(c0, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            
            torch.nn.Conv1d(c0, c1, 1, dilation=1),
            torch.nn.BatchNorm1d(c1, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            
            torch.nn.Conv1d(c1, c2, 1, dilation=1),
            torch.nn.BatchNorm1d(c2, momentum=momentum),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            
            torch.nn.Conv1d(c2, in_dim, 1, dilation=1),
            torch.nn.ReplicationPad1d((0, 0)), # no padding at end
            
            
        )
        
        self.lstm = torch.nn.LSTM(6, 50, 4, batch_first=True, dropout=0.1)
        
        

    def forward(self, us):
        # use LSTM
        

        
        us = us.permute(0, 2, 1)  # From [batch_size, sequence_length, channels] to [batch_size, channels, sequence_length]

        u = self.norm(us).transpose(1, 2)
        us = self.cnn(u).transpose(1, 2)
        
        us = us.permute(0, 2, 1)

        ys = self.lstm(us)[0][:, :, :3]
        Rots = (self.Id3 + self.gyro_Rot).expand(us.shape[0], us.shape[1], 3, 3)
        Rot_us = bbmv(Rots, us[:, :, :3]) 
        return self.gyro_std*ys + Rot_us + self.gyro_bias.view(1, 1, 3)
    
    def norm(self, us):
        return (us-self.mean_u)/self.std_u

    def set_normalized_factors(self, mean_u, std_u):
        super().set_normalized_factors(mean_u, std_u)


class CalibratedIMUNet(torch.nn.Module):
    """
    Baseline "calibrated IMU" model (paper: constant parameters).

    Predicts corrected gyro as:
        ω_hat = (I + dC) * ω_raw + b

    where dC (3x3) and b (3) are optimized by gradient descent.
    """

    def __init__(self):
        super().__init__()
        self.dC = torch.nn.Parameter(torch.zeros(3, 3))
        self.b = torch.nn.Parameter(torch.zeros(3))
        self.register_buffer("Id3", torch.eye(3))

    def forward(self, us):
        # us: [B, T, 6]
        gyro = us[:, :, :3]
        C = self.Id3 + self.dC
        Cb = C.expand(gyro.shape[0], gyro.shape[1], 3, 3)
        return bbmv(Cb, gyro) + self.b.view(1, 1, 3)
        
        
