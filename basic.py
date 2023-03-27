import collections
from webbrowser import get

import paddle
from paddle import tensor, fluid, nn
from paddle.autograd import PyLayer
from paddle.nn import functional as F
from paddle.nn.layer import Conv1D
from paddle.nn.layer.common import Linear


class BinaryQuantizer(PyLayer):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = paddle.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensor()[0]
        grad_input = grad_output
        grad_input[input >= 1] = 0
        grad_input[input <= -1] = 0
        return grad_input.clone()


class BiLinear(Linear):
    def __init__(self,
                 in_features,
                 out_features,
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        super(BiLinear, self).__init__(
            in_features,
            out_features,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name=name)
        
        self.scale_weight_init = False
        self.scale_weight = fluid.layers.create_parameter(shape=[1], dtype='float32')

    def forward(self, input):
        ba = input

        bw = self.weight
        bw = bw - bw.mean()
        
        if self.scale_weight_init == False:
            scale_weight = F.linear(ba, bw).std() / F.linear(paddle.sign(ba), paddle.sign(bw)).std()
            if paddle.isnan(scale_weight):
                scale_weight = bw.std() / paddle.sign(bw).std()
            self.scale_weight.set_value(scale_weight)            
            self.scale_weight_init = True
        
        ba = BinaryQuantizer.apply(ba)
        bw = BinaryQuantizer.apply(bw)
        bw = bw * self.scale_weight

        out = F.linear(x=ba, weight=bw, bias=self.bias, name=self.name)
        return out


class BiConv1D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, padding_mode='zeros',
                weight_attr=None, bias_attr=None, data_format="NCL"):
        super(BiConv1D, self).__init__()
        self.lin = BiLinear(in_channels, out_channels)
    
    def forward(self, x):
        N, C, L = x.shape
        x = x.transpose([0, 2, 1]).reshape([-1, C])
        x = self.lin(x).reshape([N, L, -1]).transpose([0, 2, 1])
        return x


'''
class BiConv1D(Conv1D):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, padding_mode='zeros',
                weight_attr=None, bias_attr=None, data_format="NCL"):
        super(BiConv1D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, padding_mode, weight_attr, bias_attr, data_format)
        
    def forward(self, input):
        ba = input

        bw = self.weight
        bw = bw - bw.mean()
        
        padding = 0
        if self._padding_mode != "zeros":
            ba = F.pad(ba,
                      self._reversed_padding_repeated_twice,
                      mode=self._padding_mode,
                      data_format=self._data_format)
        else:
            padding = self._padding

        if not hasattr(self, 'scale_weight'):
            scale_weight = F.conv1d(ba, bw, bias=self.bias, padding=padding, stride=self._stride, dilation=self._dilation, groups=self._groups, data_format=self._data_format).std() / \
                F.conv1d(paddle.sign(ba), paddle.sign(bw), bias=self.bias, padding=padding, stride=self._stride, dilation=self._dilation, groups=self._groups, data_format=self._data_format).std()

            if paddle.isnan(scale_weight):
                scale_weight = bw.std() / paddle.sign(bw).std()

            self.scale_weight = fluid.layers.create_parameter(shape=[1], dtype='float32', attr=fluid.initializer.Constant(value=scale_weight.detach()))
        
        binary_input_no_grad = paddle.sign(ba)
        cliped_input = paddle.clip(input, -1.0, 1.0)
        ba = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input

        binary_weights_no_grad = self.scale_weight * paddle.sign(bw)
        cliped_weights = paddle.clip(bw, -1.0, 1.0)
        bw = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        
        return F.conv1d(ba, bw, bias=self.bias, padding=padding,
                stride=self._stride, dilation=self._dilation, 
                groups=self._groups, data_format=self._data_format)
'''

def _to_bi_function(model, fp_layers=[], num=[0]):
    model.bnn = True
    for name, layer in model.named_children():
        if id(layer) in fp_layers:
            continue
        if isinstance(layer, Linear):
            if num[0] > 12:
                continue
            num[0] += 1
            new_layer = BiLinear(layer.weight.shape[0], layer.weight.shape[1],
                                layer._weight_attr, layer._bias_attr,
                                layer.name)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias
            model._sub_layers[name] = new_layer
        elif isinstance(layer, Conv1D):
            if num[0] > 12:
                continue
            num[0] += 1
            print(name, layer._in_channels, layer._out_channels, layer._kernel_size, layer._stride,
                                layer._padding, layer._dilation, layer._groups, layer._padding_mode,
                                layer._param_attr, layer._bias_attr, layer._data_format)
            new_layer = BiConv1D(layer._in_channels, layer._out_channels, layer._kernel_size, layer._stride,
                                layer._padding, layer._dilation, layer._groups, layer._padding_mode,
                                layer._param_attr, layer._bias_attr, layer._data_format)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias
            model._sub_layers[name] = new_layer
        elif isinstance(layer, nn.ReLU):
            model._sub_layers[name] = nn.Hardtanh()
        else:
            model._sub_layers[name] = _to_bi_function(layer, fp_layers, num)
    print(num)
    return model