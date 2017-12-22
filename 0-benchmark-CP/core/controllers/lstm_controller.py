from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from core.controller import Controller
from core.RNNCellBase import RNNCellBase
#my basic MyLSTMCell version

class MyLSTMCell(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(MyLSTMCell, self).__init__(input_size, hidden_size, bias)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_Parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        #print(hx)
        wx = F.linear(input, self.weight_ih, self.bias_ih)
        wh = F.linear(hx[0], self.weight_hh, self.bias_hh)
        i = F.sigmoid(wx[:, :self.hidden_size] + wh[:, :self.hidden_size])
        f = F.sigmoid(wx[:, self.hidden_size:2 * self.hidden_size] + wh[:, self.hidden_size:2 * self.hidden_size])
        g = F.tanh(wx[:, 2 * self.hidden_size:3 * self.hidden_size] + wh[:, 2 * self.hidden_size:3 * self.hidden_size])
        o = F.sigmoid(wx[:, 3 * self.hidden_size:] + wh[:, 3 * self.hidden_size:])
        c_new = f * hx[1] + i * g
        h_new = o * F.tanh(c_new)
        return h_new, c_new #(h_new, c_new)

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
