import torch
from torch import nn
from torch.nn.utils import parametrizations
import torchinfo

class FcModel(nn.Module):
    def __init__(self, dims):
        super(FcModel, self).__init__()
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_layer = nn.Sequential(
            parametrizations.weight_norm(nn.Linear(dims[0], dims[1]), dim=0),
            nn.SiLU())
        self.hidden_layers = self.create_layers(dims[1 : -1])
        self.output_layer = parametrizations.weight_norm(nn.Linear(dims[-2], dims[-1]), dim=0)

    def create_layers(self, hidden_dims):
        layers = []
        #here, dims = [in_dim, hidden_layer_dim, ... , out_dim],
        #hidden_dims = [hidden_layer_dim, ... , out_dim]
        for i in range(len(hidden_dims) - 1):
            layers.append(parametrizations.weight_norm(nn.Linear(hidden_dims[i], hidden_dims[i + 1]), dim=0))
            layers.append(nn.SiLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        out = self.output_layer(x)
        return out

if __name__ == '__main__':
    #t, x, y -> c, u, v, p
    dims = [3] + 11 * [200] + [4]
    model = FcModel(dims)
    torchinfo.summary(model, input_size=(1, 3))

        