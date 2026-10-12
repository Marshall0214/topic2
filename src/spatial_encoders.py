"""Experimental current-frame spatial adapters; original baseline stays unchanged."""
import torch
from torch import nn
from baseline import build_model

# Kinect-v2 25-joint physical links; zero-based indexing.
EDGES = [(0,1),(1,20),(20,2),(2,3),(20,4),(4,5),(5,6),(6,7),(7,21),(6,22),
         (20,8),(8,9),(9,10),(10,11),(11,23),(10,24),(0,12),(12,13),(13,14),
         (14,15),(0,16),(16,17),(17,18),(18,19)]


def unpack_features(x):
    """Input order: relative[body,joint,xyz], then backward displacement same order."""
    n, channels, t = x.shape
    if channels != 300:
        raise ValueError('Expected 300D shared skeleton features')
    relative = x[:,:150].reshape(n,2,25,3,t)
    velocity = x[:,150:].reshape(n,2,25,3,t)
    return torch.cat([relative,velocity],dim=3).permute(0,1,3,2,4).reshape(n*2,6,25,t)


class GraphLayer(nn.Module):
    def __init__(self,cin,cout):
        super().__init__()
        adjacency=torch.zeros(25,25)
        for i,j in EDGES:
            adjacency[i,j]=adjacency[j,i]=1
        adjacency/=adjacency.sum(1,keepdim=True).clamp_min(1)
        self.register_buffer('adjacency',adjacency)
        self.self_projection=nn.Conv2d(cin,cout,1)
        self.neighbor_projection=nn.Conv2d(cin,cout,1,bias=False)

    def forward(self,x):
        neighbors=torch.einsum('ij,ncjt->ncit',self.adjacency,x)
        return torch.relu(self.self_projection(x)+self.neighbor_projection(neighbors))


class SkeletonGraphEncoder(nn.Module):
    def __init__(self,width=64):
        super().__init__()
        self.layers=nn.Sequential(GraphLayer(6,32),GraphLayer(32,32))
        self.output=nn.Conv1d(2*25*32,width,1)

    def forward(self,x):
        n,_,t=x.shape
        nodes=unpack_features(x)
        # Absent body slots remain zero after bias-bearing layers. No temporal averaging.
        present=nodes.abs().sum((1,2),keepdim=True)>0
        h=self.layers(nodes)*present
        h=h.reshape(n,2,32,25,t).permute(0,1,3,2,4).reshape(n,2*25*32,t)
        return self.output(h)


def build_spatial_model(encoder='linear',stages=4,layers=10,width=64):
    # Build the exact old model FIRST, preserving identical temporal initialization
    # under the same torch seed across all adapter variants.
    model=build_model('mstcn',stages=stages,layers=layers,width=width)
    if encoder=='linear':
        return model
    if encoder=='mlp':
        model.stage1.conv_1x1=nn.Sequential(nn.Conv1d(300,287,1),nn.ReLU(),nn.Conv1d(287,width,1))
    elif encoder=='gcn':
        model.stage1.conv_1x1=SkeletonGraphEncoder(width)
    else:
        raise ValueError(encoder)
    return model
