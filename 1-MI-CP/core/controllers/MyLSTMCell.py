class MyLSTMCell(RNNCellBase):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(4 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        wx = F.linear(input, self.weight_ih, self.bias_ih)
        wh = F.linear(hx[0], self.weight_hh, self.bias_hh)
        i = F.sigmoid(wx[:, :self.hidden_size] + wh[:, :self.hidden_size])
        f = F.sigmoid(wx[:, self.hidden_size:2 * self.hidden_size] + wh[:, self.hidden_size:2 * self.hidden_size])
        g = F.tanh(wx[:, 2 * self.hidden_size:3 * self.hidden_size] + wh[:, 2 * self.hidden_size:3 * self.hidden_size])
        o = F.sigmoid(wx[:, 3 * self.hidden_size:] + wh[:, 3 * self.hidden_size:])
        c_new = f * hx[1] + i * g
        h_new = o * F.tanh(c_new)
        return h_new, (h_new, c_new)
    """A long short-term memory (LSTM) cell.

    .. math::

        \begin{array}{ll}
        i = sigmoid(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\
        f = sigmoid(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\
        g = \tanh(W_{ig} x + b_{ig} + W_{hc} h + b_{hg}) \\
        o = sigmoid(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\
        c' = f * c + i * g \\
        h' = o * \tanh(c_t) \\
        \end{array}

    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If `False`, then the layer does not use bias weights `b_ih` and `b_hh`. Default: True

    Inputs: input, (h_0, c_0)
        - **input** (batch, input_size): tensor containing input features
        - **h_0** (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
        - **c_0** (batch. hidden_size): tensor containing the initial cell state for each element in the batch.

    Outputs: h_1, c_1
        - **h_1** (batch, hidden_size): tensor containing the next hidden state for each element in the batch
        - **c_1** (batch, hidden_size): tensor containing the next cell state for each element in the batch

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape `(input_size x hidden_size)`
        weight_hh: the learnable hidden-hidden weights, of shape `(hidden_size x hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_size)`

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = Variable(torch.randn(6, 3, 10))
        >>> hx = Variable(torch.randn(3, 20))
        >>> cx = Variable(torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """



      
        