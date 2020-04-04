from enum import unique, Enum


@unique
class PHASE(Enum):
    TRAIN = 'train'
    TEST = 'test'
    EVAL = 'eval'
    PREDICT = 'predict'


@unique
class OPTIMIZER(Enum):
    SGD = 'sgc'
    ADAM = 'adam'
    RMSPROP = 'rmsprop'
    MOMENTUM = 'momentum'


@unique
class LOSS(Enum):
    MSE = 'mse'
    CE = 'ce'


