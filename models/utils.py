import torch
import torch.nn as nn

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    ''' Orthogonal weight initialization old way

    :param m: The NN model
    '''
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


def initialize_parameters2(m):
    ''' Orthogonal weight initialization new way

    :param m: The NN model
    '''

    def init_(module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)

    def conv_init_(m):
        init_(m, nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

    def linear_init_(m):
        init_(m, nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0))

    def recurrent_init_(m):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    classname = m.__class__.__name__

    if classname.find("Conv2d") != -1:
        conv_init_(m)
    elif classname.find("Linear") != -1:
        linear_init_(m)
    elif classname.find("LSTMCell") != -1 or classname.find("GRUCell") != -1:
        recurrent_init_(m)


def initialize_parameters_ec(m):
    '''Episodic curiosity RNetwork weight initialization
    :param m: The NN model
    '''
    classname = m.__class__.__name__

    if classname.find("Conv2d") != -1:
        nn.init.kaiming_normal(m.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.constant(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal(m.weight.data, mode='fan_in', nonlinearity='relu')
        nn.init.constant(m.bias.data, 0)
