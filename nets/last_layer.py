import torch
import torch.nn as nn
from torch.nn import Parameter
def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class ArcFace(nn.Module):
    def __init__(self, input_size, output_size):
        super(ArcFace, self).__init__()
        self.input = input_size
        self.output_size = output_size
        self.kernel = Parameter(torch.Tensor(input_size,output_size))
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
    def forward(self, inputs):
        # shape inputs: bs x channels x width x height
        bs = inputs.size[0]



