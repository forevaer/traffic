from config import config, enum
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
            image = item['image'].to(device)
            classify = item['classify'].to(device)
            # =====
            batch_count = image.size(0)
            optimizer.zero_grad()
            pre_softmax = model(image)
            pre_classify = pre_softmax.argmax(dim=1, keepdim=True).view(-1)
            _loss = loss_func(pre_softmax, classify.view(-1))
            # loss
            positive_sample = pre_classify == classify
            batch_correct_count = torch.sum(positive_sample)
            if config.weight_switch_on.value:
                # 对预测错误的，采取更大权值激励，以提高泛化能力<三通道中，对颜色敏感，降低对颜色的敏感>
                negative_sample = positive_sample == False
                batch_error_count = torch.sum(negative_sample)
                positive_loss = 0
                negative_loss = 0
                if batch_correct_count > 0:
                    positive_loss = loss_func(pre_softmax[positive_sample], classify[positive_sample].view(-1)) * batch_correct_count * config.positive_weight
                if batch_error_count > 0:
                    negative_loss = loss_func(pre_softmax[negative_sample], classify[negative_sample].view(-1)) * batch_error_count * config.negative_weight
                batch_loss = positive_loss + negative_loss
            else:
                batch_loss = loss_func(pre_softmax, classify.view(-1)) * batch_count * config.normal_weight
            batch_acc = 1.0 * batch_correct_count / batch_count
            batch_loss.backward()
            optimizer.step()
            # ===
            trained_count += batch_count
            correct_count += batch_correct_count
            loss += batch_loss
            avg_correct_acc = 1.0 * correct_count / trained_count
            avg_loss = 1.0 * loss / (idx + 1)
            # ===
            if draw is not None:
                draw(config.train_epoch, epoch, trained_count, count, batch_loss, avg_loss, batch_acc, avg_correct_acc, (epoch == (config.train_epoch - 1)) and (trained_count == count))
            if idx % config.log_interval == 0:
                log(epoch, idx, trained_count, count, batch_acc, avg_correct_acc, batch_loss, avg_loss)
            if idx % config.save_model_interval == 0:
                model_path = config.model_path()
                model_dir = os.path.dirname(model_path)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(model.state_dict(), model_path)
