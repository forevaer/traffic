from config import config
from torch.nn import Module
from torch.nn import Conv2d, PReLU, MaxPool2d, BatchNorm2d, AvgPool2d, Linear, BatchNorm1d
from torch.nn.functional import dropout


class Net(Module):

    def __init__(self):
        super(Net, self).__init__()
        # 3->10; (200 - 5)/2 + 1 = 98
        # 三通道，由于样本的局限，对颜色较为敏感，后续采用灰度图像，降低对颜色依赖
        # self.layer_1_conv = Conv2d(1, 10, 5, 2, 0)
        # 3->10; (200 - 5)/2 + 1 = 98
        self.layer_1_conv = Conv2d(config.input_channel(), 10, 5, 2, 0)
        self.layer_1_bn = BatchNorm2d(10)
        self.layer_1_relu = PReLU()
        # 10->30; 98 - 3 + 1 = 96
        self.layer_2_conv_1 = Conv2d(10, 30, 3, 1, 0)
        # 30->30; 96 - 3 + 1 = 94
        self.layer_2_conv_2 = Conv2d(30, 30, 3, 1, 0)
        self.layer_2_bn = BatchNorm2d(30)
        self.layer_2_relu = PReLU()
        # 30->30; 94/2 = 47
        self.layer_2_pool = MaxPool2d(2, 2)
        # 30->90; 47 - 3 + 1 = 45
        self.layer_3_conv_1 = Conv2d(30, 90, 3, 1, 0)
        # 90->90; 45 - 3 + 1 = 43
        self.layer_3_conv_2 = Conv2d(90, 90, 3, 1, 0)
        self.layer_3_bn = BatchNorm2d(90)
        self.layer_3_relu = PReLU()
        # 90->90; 43 / 2 = 21
        self.layer_3_pool = MaxPool2d(2, 2)

        # 90->100; 21 - 3 + 1 = 19
        self.layer_4_conv_1 = Conv2d(90, 100, 3, 1, 0)
        # 100->100; 19 - 3 + 1 = 17
        self.layer_4_conv_2 = Conv2d(100, 100, 3, 1, 0)
        self.layer_4_bn = BatchNorm2d(100)
        self.layer_4_relu = PReLU()
        # 100->100; (17 - 4) / 4 + 1 = 4
        self.layer_4_pool = AvgPool2d(4, 4)

        self.fc_1 = Linear(4 * 4 * 100, 160)
        self.fc_1_bn = BatchNorm1d(160)
        self.fc_1_relu = PReLU()
        self.fc_2 = Linear(160, 100)
        self.fc_2_bn = BatchNorm1d(100)
        self.fc_2_relu = PReLU()
        self.fc_3 = Linear(100, config.classify_count)

    def forward(self, x):
        # layer 1
        layer_1_conv = self.layer_1_conv(x)
        layer_1_bn = self.layer_1_bn(layer_1_conv)
        layer_1_relu = self.layer_1_relu(layer_1_bn)
        # layer 2
        layer_2_conv = self.layer_2_conv_1(layer_1_relu)
        layer_2_conv = self.layer_2_conv_2(layer_2_conv)
        layer_2_bn = self.layer_2_bn(layer_2_conv)
        layer_2_relu = self.layer_2_relu(layer_2_bn)
        layer_2_pool = self.layer_2_pool(layer_2_relu)
        # layer 3
        layer_3_conv = self.layer_3_conv_1(layer_2_pool)
        layer_3_conv = self.layer_3_conv_2(layer_3_conv)
        layer_3_bn = self.layer_3_bn(layer_3_conv)
        layer_3_relu = self.layer_3_relu(layer_3_bn)
        layer_3_pool = self.layer_3_pool(layer_3_relu)
        # layer 4
        layer_4_conv = self.layer_4_conv_1(layer_3_pool)
        layer_4_conv = self.layer_4_conv_2(layer_4_conv)
        layer_4_relu = self.layer_4_relu(layer_4_conv)
        layer_4_pool = self.layer_4_pool(layer_4_relu)
        # fc 1
        fc_input = layer_4_pool.view(-1, 4 * 4 * 100)
        fc_1 = dropout(fc_input, training=self.training)
        fc_1 = self.fc_1(fc_1)
        fc_1_bn = self.fc_1_bn(fc_1)
        fc_1_relu = self.fc_1_relu(fc_1_bn)
        # fc 2
        fc_2 = dropout(fc_1_relu, training=self.training)
        fc_2 = self.fc_2(fc_2)
        fc_2_bn = self.fc_2_bn(fc_2)
        fc_2_relu = self.fc_2_relu(fc_2_bn)
        # fc 3
        fc_3 = dropout(fc_2_relu, training=self.training)
        fc_3 = self.fc_3(fc_3)

        return fc_3



