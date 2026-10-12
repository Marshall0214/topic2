import torch
import torch.nn as nn

class featureEncoder(nn.Module): 

    def __init__(self, dim = 64, eps_std=0.00005, mu_size=None):
        super(featureEncoder, self).__init__()
        self.mu_enc = nn.Linear(dim, dim)
        self.var_enc = nn.Sequential(nn.Linear(dim, dim), 
                                     nn.Dropout(0.2), 
                                     nn.ReLU(), 
                                     nn.Linear(dim, dim))
        self.relu = nn.ReLU()
        self.eps_std = eps_std

    def sample_gaussian_tensors(self, mu, logsigma, num_samples):
        eps = torch.normal(0, self.eps_std, size=(num_samples, mu.size(0), mu.size(1), mu.size(2))).to(mu.device)
        samples = torch.einsum('nbtd,btd->nbtd', [eps, torch.exp(.5*logsigma)]) 
        samples = samples.permute(1,0,2,3)
        samples = samples + mu.unsqueeze(1).repeat(1,num_samples,1,1)
        return samples.mean(dim=1) 

    def forward(self, f, num_samples, is_training): 
        f = f.permute((0, 2, 1))
        mu = self.mu_enc(f)
        var = self.relu(self.var_enc(f))

        if is_training:
            embedding = self.sample_gaussian_tensors(mu, var, num_samples)
            # print(mu.shape, embedding.shape, var.shape)
            # exit()
            return embedding.permute(0, 2, 1)
        else:
           return mu.permute(0, 2, 1)
    
