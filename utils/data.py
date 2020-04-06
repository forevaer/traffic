import os
import torch
from config import config, trans
from config.enum import PHASE
from entity.dataset import TrafficDataSet, TrafficCSVLoader
from torch.utils.data import DataLoader


def get_loader():
    batch_size = 1
    shuffle = False
    predict = False
    transform = trans.test_transform
    if config.phase is PHASE.TRAIN:
        load_path = config.train_csv
        batch_size = config.train_batch
        shuffle = config.train_shuffle
        transform = trans.train_transform
    elif config.phase is PHASE.TEST:
        load_path = config.test_csv
        batch_size = config.test_batch
        shuffle = config.test_shuffle
    elif config.phase is PHASE.PREDICT:
        load_path = config.predict_images
        predict = True
    else:
        raise Exception(f'unSupport phase : {config.phase}')
    csv_loader = TrafficCSVLoader(load_path, predict)
    dataset = TrafficDataSet(csv_loader, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    count = len(dataset)
    return loader, count


def prepare_model(model: torch.nn.Module):
    model_path = config.model_path()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    return model


if __name__ == '__main__':
    for _loader in get_loader():
        for _ in enumerate(_loader):
            pass
