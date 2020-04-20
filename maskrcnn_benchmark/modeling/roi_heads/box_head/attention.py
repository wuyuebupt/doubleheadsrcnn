import torch
from torch import nn
from torch.nn import functional as F


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)



class FPNFFConv(nn.Module):
    def __init__(self, in_channels):
        super(FPNFFConv, self).__init__()

        inter_channels = in_channels // 4
        out_channels = in_channels

        self.relu = nn.ReLU(inplace=True)
        ## top
        self.bottleneck = nn.Sequential(
                             nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.BatchNorm2d(inter_channels),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
                             nn.BatchNorm2d(inter_channels),
                             nn.ReLU(inplace=True),
                             nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                             nn.BatchNorm2d(out_channels)
          )


    def forward(self, x):
        identity = x
        ## bottom
        out = self.bottleneck(x)
        ## residual
        out1 = out + identity
        out1 = self.relu(out1)

        return out1


### group non local
class _NonLocalBlockND_Group(nn.Module):
    def __init__(self, in_channels, num_group, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, relu_layer=True, use_softmax=True, use_ffconv=True, use_attention=True):
        super(_NonLocalBlockND_Group, self).__init__()

        assert dimension in [1, 2, 3]
        assert dimension == 2
        assert num_group in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.num_group = num_group

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        ## inner channels are divided by num of groups
        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        bn = nn.BatchNorm2d

        self.relu_layer = relu_layer
        self.relu = nn.ReLU(inplace=True)

        self.use_softmax = use_softmax

        self.use_ffconv = use_ffconv
        self.use_attention = use_attention

        if self.use_softmax:
            self.softmax = nn.Softmax(dim=2)
     

        assert self.num_group <= self.inter_channels

        if self.use_attention:
            self.inter_channels_group = self.inter_channels // self.num_group
            print (self.inter_channels_group)

            self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)

            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            assert sub_sample==False
            if sub_sample:
                self.g = nn.Sequential(self.g, max_pool_layer)
                self.phi = nn.Sequential(self.phi, max_pool_layer)


            self.W = nn.Sequential(conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0))

            ## BN first then RELU
            if bn_layer:
                self.W.add_module(
                    'bn', bn(self.in_channels)
                )


            ## init the weights
            nn.init.constant_(self.W[0].weight, 0)
            nn.init.constant_(self.W[0].bias, 0)


        if self.use_ffconv:
            self.ffconv = FPNFFConv(self.in_channels)



    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        if self.use_attention:
            batch_size = x.size(0)
            g_x = self.g(x).view(batch_size, self.inter_channels, -1)
            g_x = g_x.permute(0, 2, 1)
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

            if self.num_group == 1:
                f = torch.matmul(theta_x, phi_x)

                if self.use_softmax == True:
                    f_div_C = self.softmax(f)
                else:
                    N = f.size(-1)
                    f_div_C = f / N

                yy = torch.matmul(f_div_C, g_x)
                yy = yy.permute(0, 2, 1).contiguous()

                yy = yy.view(batch_size, self.inter_channels, *x.size()[2:])
                W_y = self.W(yy)
            else:    
                g_xs = torch.split(g_x, self.inter_channels_group, dim=2)
                theta_xs = torch.split(theta_x, self.inter_channels_group, dim=2) 
                phi_xs = torch.split(phi_x, self.inter_channels_group, dim=1)
                y_group = []
                for gx, tx, px in zip(g_xs, theta_xs, phi_xs):
                    f = torch.matmul(tx, px)

                    if self.use_softmax == True:
                        f_div_C = self.softmax(f)
                    else:
                        N = f.size(-1)
                        f_div_C = f / N

                    yy = torch.matmul(f_div_C, gx)
                    yy = yy.permute(0, 2, 1).contiguous()
                    y_group.append(yy)

                y_out = torch.cat(y_group, dim=1)
                y_out = y_out.view(batch_size, self.inter_channels, *x.size()[2:])
                W_y = self.W(y_out)

            z = W_y + x

            ## relu after residual
            if self.relu_layer:
                z = self.relu(z)
        else:
            z = x

        ## add ffconv
        if self.use_ffconv:
            zz = self.ffconv(z)
        else:
            zz = z

        return zz


class NONLocalBlock2D_Group(_NonLocalBlockND_Group):
    def __init__(self, in_channels, num_group=1, inter_channels=None, sub_sample=True, bn_layer=True, relu_layer=True, use_softmax=True, use_ffconv=True, use_attention=True):
        super(NONLocalBlock2D_Group, self).__init__(in_channels,
                                              num_group=num_group,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, relu_layer=relu_layer, use_softmax=use_softmax, use_ffconv=use_ffconv, use_attention=use_attention)


## original non local
class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    import torch

    for (sub_sample, bn_layer) in [(True, True), (False, False), (True, False), (False, True)]:
        img = torch.zeros(2, 3, 20)
        net = NONLocalBlock1D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.zeros(2, 3, 20, 20)
        net = NONLocalBlock2D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())

        img = torch.randn(2, 3, 8, 20, 20)
        net = NONLocalBlock3D(3, sub_sample=sub_sample, bn_layer=bn_layer)
        out = net(img)
        print(out.size())


