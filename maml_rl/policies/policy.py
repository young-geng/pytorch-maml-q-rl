import torch
import torch.nn as nn

from collections import OrderedDict

def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def update_params(self, loss, step_size=0.5, first_order=False, params=None):
        """Apply one step of gradient descent on the loss function `loss`, with 
        step-size `step_size`, and returns the updated parameters of the neural 
        network.
        """
        if params is None:
            parameters = self.parameters()
            named_parameters = self.named_parameters()
        else:
            parameters = params.values()
            named_parameters = params
        grads = torch.autograd.grad(loss, parameters,
            create_graph=not first_order)
        updated_params = OrderedDict()
        for (name, param), grad in zip(named_parameters, grads):
            updated_params[name] = param - step_size * grad

        return updated_params
