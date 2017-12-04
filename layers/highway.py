import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    '''
        References
        - Highway Networks
          http://arxiv.org/abs/1505.00387v2
    '''
    def __init__(self, in_size, out_size, n_layers=3, act=F.relu, final_act=F.softmax):
        super(Highway, self).__init__()

        self.n_layers = n_layers
        self.act = act
        self.final_act = final_act

        self.normal_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(n_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(n_layers)])

        self.last_layer = nn.Linear(in_size, out_size)

    def forward(self, x):
        for i in range(self.n_layers):
            normal_layer_ret = self.act(self.normal_layer[i](x))
            gate = F.sigmoid(self.gate_layer[i](x))

            x = gate * normal_layer_ret + (1 - gate) * x

        x = self.final_act(self.last_layer(x))

        return x
