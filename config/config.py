import torch
from config.enum import PHASE, OPTIMIZER, LOSS, IMAGE, SWITCH
from torch import optim, nn

draw = True
draw_count = 30
# 对于三色通道，对于预测错误样本损失通过权值放大，达到对颜色敏感的降低，后采用灰度图直接推理，摒弃颜色影响
image_format = IMAGE.RGB
weight_switch_on = SWITCH.OFF
positive_weight = 10
negative_weight = 30
normal_weight = 10
#
resize_H = 200
resize_W = resize_H
resize = resize_H, resize_W
train_csv = "../train.csv"
test_csv = '../test.csv'
predict_images = [
    '../data/test/00045/00045_0.png',
    '../data/test/00045/00045_1.png',
    '../data/test/00001/00001_2.png',
    '../data/test/00004/00004_0.png',
    '../data/test/00021/00021_0.png',
    '../data/test/00027/00027_1.png',
    '../data/test/00055/00055_0.png',
    '../data/test/00038/00038_0.png',
    '../data/test/00040/00040_0.png',
    '../data/test/00042/00042_0.png',
    '../data/test/00042/00042_1.png',
]
# ====
train_batch = 200
train_shuffle = True
train_epoch = 1000
test_epoch = 10
test_batch = 20
test_shuffle = True
# ====
phase = PHASE.TRAIN
# ====
learn_rate = 0.0001
momentum = 0.9
default_optimizer = OPTIMIZER.MOMENTUM
default_loss = LOSS.CE
# ===
log_interval = 1
save_model_interval = 2
# ===
classify_count = 62


def device():
    if torch.cuda.is_available():
        return torch.device('gpu')
    return torch.device('cpu')


def optimizer(model: torch.nn.Module, op: OPTIMIZER = default_optimizer):
    params = model.parameters()
    if op is OPTIMIZER.MOMENTUM:
        return optim.SGD(params, lr=learn_rate, momentum=momentum)
    if op is OPTIMIZER.ADAM:
        return optim.Adam(params, lr=learn_rate)
    if op is OPTIMIZER.RMSPROP:
        return optim.RMSprop(params, lr=learn_rate)
    return optim.SGD(params, lr=learn_rate)


def loss(loss_type: LOSS = default_loss):
    if loss_type is LOSS.CE:
        return nn.CrossEntropyLoss()
    return nn.MSELoss()


def input_channel(_image_format: IMAGE = image_format):
    return len(_image_format.value)


def model_path(_image_format: IMAGE = image_format):
    return f'../pts/model_{_image_format.value}_{weight_switch_on.name}.pt'
