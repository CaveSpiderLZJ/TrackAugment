import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    
    
    def __init__(self) -> None:
        super().__init__()
        self.kernel_size = 5
        self.embed_length = 100
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=6,
            kernel_size=self.kernel_size, padding=2, padding_mode='replicate', stride=1)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.linear1 = nn.Linear(in_features=300, out_features=self.embed_length)
        self.linear2 = nn.Linear(in_features=self.embed_length, out_features=6*self.embed_length)
        self.linear3 = nn.Linear(in_features=self.embed_length, out_features=200)
        
    
    def forward(self, x, generation=False) -> torch.Tensor:
        if generation == False:
            x = self.conv1(x)
            x = self.pool(x)
            x = torch.reshape(x, (x.shape[0], -1))
            neck = self.linear1(x)       
            x = self.linear2(neck)
            x = torch.reshape(x, (x.shape[0], 6, self.embed_length))
            x = self.linear3(x)
            return x, neck
        else:
            x = self.linear2(x)
            x = torch.reshape(x, (x.shape[0], 6, self.embed_length))
            x = self.linear3(x)
            return x


class VAE(nn.Module):
    
    
    def __init__(self):
        super().__init__()
        # config
        in_channels = 6
        window_length = 200
        feature_channels = 12
        hid_length = 100
        z_length = 50
        # feature extraction
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=feature_channels,
            kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.contract = nn.MaxPool1d(kernel_size=4)
        self.flatten = nn.Flatten()
        # feature to hid
        self.feature2hid = nn.Linear(in_features=50*feature_channels, out_features=hid_length)
        # hid to probability distribution of z
        self.hid2mu = nn.Linear(in_features=hid_length, out_features=z_length)
        self.hid2sigma = nn.Linear(in_features=hid_length, out_features=z_length)
        # z to hid
        self.z2hid = nn.Linear(in_features=z_length, out_features=hid_length)
        # # hid to feature
        # self.hid2feature = nn.Linear(in_features=hid_length, out_features=600)
        # # reconstruction
        # self.expand = nn.Linear(in_features=100, out_features=window_length)
        # hid to data
        self.hid2data1 = nn.Linear(in_features=hid_length, out_features=300)
        self.hid2data2 = nn.Linear(in_features=300, out_features=in_channels*window_length)
        

    def encode(self, x):
        x = self.conv1(x)                   # (12, 200)
        x = self.contract(x)                # (12, 50)
        x = self.flatten(x)                 # (600,)
        x = F.silu(self.feature2hid(x))     # (100,)
        mu = self.hid2mu(x)                 # (50,)
        sigma = self.hid2sigma(x)           # (50,)
        return mu, sigma


    def decode(self, x):
        x = F.silu(self.z2hid(x))           # (100,)
        x = F.silu(self.hid2data1(x))        # (300,)
        x = self.hid2data2(x)               # (1200,)
        x = torch.reshape(x, (x.shape[0], 6, 200))
        return x
        # x = F.silu(self.hid2feature(x))     # (600,)
        # x = torch.reshape(x, (x.shape[0], 6, 100))  # (6, 100)
        # x = self.expand(x)                  # (6, 200)
        # return x


    def forward(self, x):
        mu, sigma = self.encode(x)
        # Sample from latent distribution from encoder
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        x = self.decode(z)
        return x, mu, sigma
    
    
class VAE2(nn.Module):
    
    
    def __init__(self):
        super().__init__()
        # config
        in_channels = 6
        window_length = 200
        feature_channels = 12
        z_length = 100
        self.z_length = z_length
        # feature extraction
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=feature_channels,
            kernel_size=5, stride=1, padding=2, padding_mode='replicate')
        self.contract = nn.MaxPool1d(kernel_size=4)
        self.flatten = nn.Flatten()
        # hid to probability distribution of z
        self.hid2mu = nn.Linear(in_features=50*feature_channels, out_features=z_length)
        self.hid2sigma = nn.Linear(in_features=50*feature_channels, out_features=z_length)
        # z to hid
        self.z2hid = nn.Linear(in_features=z_length, out_features=6*z_length)
        self.expand = nn.Linear(in_features=z_length, out_features=window_length)
        

    def encode(self, x):
        x = self.conv1(x)                   # (6, 200)
        x = self.contract(x)                # (6, 50)
        x = self.flatten(x)                 # (300,)
        mu = self.hid2mu(x)                 # (100,)
        sigma = self.hid2sigma(x)           # (100,)
        return mu, sigma


    def decode(self, x):
        x = self.z2hid(x)                   # (600,)
        x = torch.reshape(x,(x.shape[0],6,self.z_length)) # (6, 100)
        x = self.expand(x)                  # (6, 200)
        return x


    def forward(self, x):
        mu, sigma = self.encode(x)
        # Sample from latent distribution from encoder
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        x = self.decode(z)
        return x, mu, sigma
    
    
class ConditionalGenerator(nn.Module):
    
    
    def __init__(self):
        super(ConditionalGenerator, self).__init__()
        self.linear1 = nn.Linear(in_features=104, out_features=600)
        self.act1 = nn.SiLU()
        self.expand = nn.Linear(in_features=100, out_features=200)
    
    
    def forward(self, x, y):
        x = torch.concatenate([x, y], dim=1)
        x = self.act1(self.linear1(x))
        x = torch.reshape(x, (x.shape[0], 6, 100))
        x = self.expand(x)
        return x
    

class ConditionalDiscriminator(nn.Module):
    
    
    def __init__(self):
        super(ConditionalDiscriminator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=12,
            kernel_size=5, stride=1, padding=2, padding_mode='replicate')
        self.contract = nn.MaxPool1d(kernel_size=4)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=604, out_features=200)
        self.act1 = nn.SiLU()
        self.linear2 = nn.Linear(in_features=200, out_features=100)
        self.act2 = nn.SiLU()
        self.linear3 = nn.Linear(in_features=100, out_features=50)
        self.act3 = nn.SiLU()
        self.linear4 = nn.Linear(in_features=50, out_features=1)
        self.act4 = nn.Sigmoid()
        
        
    def forward(self, x, y):
        x = self.conv1(x)
        x = self.contract(x)
        x = self.flatten(x)
        x = torch.concatenate([x, y], dim=1)
        x = self.act1(self.linear1(x))
        x = self.act2(self.linear2(x))
        x = self.act3(self.linear3(x))
        x = self.act4(self.linear4(x))
        return x
    
    
class ConditionalGenerator2(nn.Module):
    
    
    def __init__(self):
        super(ConditionalGenerator2, self).__init__()
        self.linear1 = nn.Linear(in_features=50, out_features=100)
        self.act1 = nn.SiLU()
        self.linear2 = nn.Linear(in_features=100, out_features=200)
        self.act2 = nn.SiLU()
        self.convt1 = nn.ConvTranspose1d(in_channels=8, out_channels=16,
            kernel_size=4, stride=2, padding=1)
        self.act3 = nn.SiLU()
        self.convt2 = nn.ConvTranspose1d(in_channels=16, out_channels=32,
            kernel_size=4, stride=2, padding=1)
        self.act4 = nn.SiLU()
        self.convt3 = nn.ConvTranspose1d(in_channels=32, out_channels=6,
            kernel_size=4, stride=2, padding=1)
    
    
    def forward(self, x, y):
        x = torch.concatenate([x, y], dim=1)
        x = self.act1(self.linear1(x))
        x = self.act2(self.linear2(x))
        x = torch.reshape(x, (x.shape[0], 8, 25))
        x = self.act3(self.convt1(x))
        x = self.act4(self.convt2(x))
        x = self.convt3(x)
        return x

    
class ConditionalDiscriminator2(nn.Module):
    
    
    def __init__(self):
        super(ConditionalDiscriminator2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=6, out_channels=1, kernel_size=4,
            stride=2, padding=1, padding_mode='replicate')
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=104, out_features=50)
        self.act4 = nn.SiLU()
        self.linear2 = nn.Linear(in_features=50, out_features=1)
        self.act5 = nn.Sigmoid()
        
        
    def forward(self, x, y):
        x = self.conv1(x)
        x = self.flatten(x)
        x = torch.concatenate([x, y], dim=1)
        x = self.act4(self.linear1(x))
        x = self.act5(self.linear2(x))
        return x
    
    
def calc_model_parameters():
    model = VAE2()
    params = model.parameters()
    total = 0
    for item in params:
        print(item.shape, np.prod(item.shape))
        total += np.prod(item.shape)
    print(f'### total: {total // 1000} k')

    
if __name__ == '__main__':
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    model = ConditionalDiscriminator2()
    x = torch.randn(10, 6, 200)
    y = torch.randn(10, 4)
    output = model(x, y)
    print(output.shape)
