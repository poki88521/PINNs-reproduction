from torch import nn
import torchinfo

class FeedForwardNet(nn.Module):
    def __init__(self, dims):
        super(FeedForwardNet, self).__init__()
        self.dims = dims
        self.layers = self.create_layers(dims)
        self.activation = nn.Tanh()

    def create_layers(self, dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            nn.init.xavier_normal_(layers[i].weight.data, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(layers[i].bias.data)
        return nn.ModuleList(layers)

    def forward(self, x):
        for i in range(len(self.layers) - 2):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x)


class AttentionNet(FeedForwardNet):
    def __init__(self, dims):
        super(AttentionNet, self).__init__(dims)
        self.attentions = nn.ModuleList([nn.Linear(dims[0], dims[1]), nn.Linear(dims[0], dims[1])])
        nn.init.xavier_normal_(self.attentions[0].weight.data, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.attentions[0].bias.data)
        nn.init.xavier_normal_(self.attentions[1].weight.data, gain=nn.init.calculate_gain('tanh'))
        nn.init.zeros_(self.attentions[1].bias.data)

    def forward(self, x):
        encoders = [self.activation(self.attentions[0](x)), self.activation(self.attentions[1](x))]
        a = self.activation(self.layers[0](x))
        a = a * encoders[0] + (1 - a) * encoders[1]
        for i in range(1, len(self.layers) - 2):
            a = self.activation(self.layers[i](a))
            a = a * encoders[0] + (1 - a) * encoders[1]
        return self.layers[-1](a)


if __name__ == '__main__':
    dims = [2] + 5 * [70] + [1]
    model = FeedForwardNet(dims)
    torchinfo.summary(model, input_size=(1, 2))

