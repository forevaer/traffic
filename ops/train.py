from config import config
from torch import nn
import torch
import os
from utils.log import log
from utils.image import draw_func


def TRAIN(model: nn.Module, device, loader, count):
    loss_func = config.loss()
    optimizer = config.optimizer(model)
    draw = draw_func() if config.draw else None
    model.train()
    for epoch in range(config.train_epoch):
        trained_count = 0
        correct_count = 0
        loss = 0
        for idx, item in enumerate(loader):
            image = item['image']
            classify = item['classify']
            # =====
            image = image.to(device)
            classify = classify.to(device)
            # =====
            batch_count = image.size(0)
            optimizer.zero_grad()
            pre_softmax = model(image)
            pre_classify = pre_softmax.argmax(dim=1, keepdim=True).view(-1)
            _loss = loss_func(pre_softmax, classify.view(-1))
            batch_loss = _loss * batch_count
            batch_compare_result = pre_classify == classify
            batch_correct_count = torch.sum(batch_compare_result)
            batch_loss.backward()
            optimizer.step()
            trained_count += batch_count
            correct_count += batch_correct_count
            loss += batch_loss
            batch_acc = 1.0 * batch_correct_count / batch_count
            avg_correct_acc = 1.0 * correct_count / trained_count
            avg_loss = 1.0 * loss / (idx + 1)
            if draw is not None:
                draw(batch_loss, avg_loss, batch_acc, avg_correct_acc, epoch == (config.train_epoch - 1))
            if idx % config.log_interval == 0:
                log(epoch, idx, trained_count, count, batch_acc, avg_correct_acc, batch_loss, avg_loss)
            if idx % config.save_model_interval == 0:
                model_dir = os.path.dirname(config.model_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(model.state_dict(), config.model_path)
