from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

from core.controller import Controller
from core.RNNCellBase import RNNCellBase
# MyLSTMCell - peephole version 

class MyLSTMCell(nn.LSTMCell): #peephole
    def __init__(self, input_size, hidden_size, bias=True):
        super(MyLSTMCell, self).__init__(input_size, hidden_size, bias)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_ch = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_ch = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            self.register_parameter('bias_ch',None)
        self.register_buffer('wc_blank', torch.zeros(hidden_size))    
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        h, c = hx
        wx = F.linear(input, self.weight_ih, self.bias_ih)
        wh = F.linear(hx[0], self.weight_hh, self.bias_hh)
        wc = F.linear(c, self.weight_ch, self.bias_ch)
        wxhc = wx + wh + torch.cat((wc[:,:2*self.hidden_size], Variable(self.wc_blank).expand_as(h),wc[:,2*self.hidden_size:]),1)
        i = F.sigmoid(wxhc[:,:self.hidden_size])
        f = F.sigmoid(wxhc[:,self.hidden_size:2*self.hidden_size])
        g = F.tanh(wxhc[:, 2*self.hidden_size:3 * self.hidden_size])
        o = F.sigmoid(wxhc[:, 3*self.hidden_size:])

        c = f*c +i*g
        h = o*F.tanh(c)
        return h, c

class LSTMController(Controller):
    def __init__(self, args):
        super(LSTMController, self).__init__(args)

        # build model
        self.in_2_hid = MyLSTMCell(self.input_dim + self.read_vec_dim, self.hidden_dim, 1)
        #nn.LSTMCell
        self._reset()

    def _init_weights(self):
        pass

    def forward(self, input_vb, read_vec_vb):
        self.lstm_hidden_vb = self.in_2_hid(torch.cat((input_vb.contiguous().view(-1, self.input_dim),
                                                       read_vec_vb.contiguous().view(-1, self.read_vec_dim)), 1),
                                            self.lstm_hidden_vb)

        # we clip the controller states here
        #print('first element type',type(self.lstm_hidden_vb[0]))
        #print('entire part type',type(self.lstm_hidden_vb))
        self.lstm_hidden_vb[0].clamp(min=-self.clip_value, max=self.clip_value)
        #print("self.lstm_hidden_vb[1]",self.lstm_hidden_vb[1])
        self.lstm_hidden_vb[1].clamp(min=-self.clip_value, max=self.clip_value) #[1]
        return self.lstm_hidden_vb[0]