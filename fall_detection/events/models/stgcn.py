import torch
import torch.nn as nn
import torch.nn.functional as F

_EDGES = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)
]

def normalized_adjacency(num_nodes=17):
    A = torch.eye(num_nodes)
    for i,j in _EDGES:
        A[i,j] = 1.0
        A[j,i] = 1.0
    deg = A.sum(1)
    deg_inv_sqrt = torch.pow(deg + 1e-6, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return (D_inv_sqrt @ A @ D_inv_sqrt).float()

class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, A, stride=1, t_kernel=9, dropout=0.0):
        super().__init__()
        self.A = A  # [K,V,V]
        k = self.A.shape[0]
        pad = (t_kernel - 1) // 2
        self.gcn = nn.Conv2d(c_in, c_out * k, kernel_size=1, bias=False)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=(t_kernel,1),
                      stride=(stride,1), padding=(pad,0), bias=False),
            nn.BatchNorm2d(c_out),
            nn.Dropout(dropout)
        )
        self.down = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=(stride,1), bias=False),
                nn.BatchNorm2d(c_out)
            )

    def forward(self, x):  # x:[B,C,T,V]
        B, C, T, V = x.shape
        y = self.gcn(x)                 # [B, C*k, T, V]
        k = self.A.shape[0]
        y = y.view(B, -1, k, T, V)      # [B, C_out, K, T, V]
        outs = []
        for i in range(k):
            Ai = self.A[i]              # [V,V]
            outs.append(torch.einsum('bctv,vw->bctw', y[:,:,i], Ai))
        y = sum(outs)                   # [B, C_out, T, V]
        y = self.tcn(y)

        res = x if self.down is None else self.down(x)
        return F.relu(y + res, inplace=True)

class STGCN(nn.Module):
    def __init__(self, in_ch=3, num_class=2, num_nodes=17, t_kernel=9, dropout=0.25):
        super().__init__()
        A = normalized_adjacency(num_nodes)
        A_stack = torch.stack([torch.eye(num_nodes), A, A.t()], dim=0)  # [3,V,V]
        self.register_buffer('A', A_stack)

        self.data_bn = nn.BatchNorm1d(in_ch * num_nodes)
        self.layers = nn.ModuleList([
            STGCNBlock(in_ch,   64, self.A, stride=1, t_kernel=t_kernel, dropout=dropout),
            STGCNBlock(64,     128, self.A, stride=2, t_kernel=t_kernel, dropout=dropout),
            STGCNBlock(128,    128, self.A, stride=1, t_kernel=t_kernel, dropout=dropout),
            STGCNBlock(128,    256, self.A, stride=2, t_kernel=t_kernel, dropout=dropout),
            STGCNBlock(256,    256, self.A, stride=1, t_kernel=t_kernel, dropout=dropout),
        ])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc   = nn.Linear(256, num_class)

    def forward(self, x):   # x:[B,3,T,17]
        B,C,T,V = x.shape
        x = x.permute(0,3,1,2).contiguous().view(B, V*C, T)  # [B, V*C, T]
        x = self.data_bn(x).view(B, V, C, T).permute(0,2,3,1).contiguous()  # [B,C,T,V]
        for blk in self.layers:
            x = blk(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
