import torch
from config import config
from utils.log import log


def TEST(model, device, loader, count):
    model.eval()
    loss_func = config.loss()
    optimizer = config.optimizer(model)
    for epoch in range(config.test_epoch):
        tested_count = 0
        correct_count = 0
        loss = 0
        for idx, item in enumerate(loader):
            image = item['image'].to(device)
            classify = item['classify'].to(device)
            batch_count = image.size(0)
            optimizer.zero_grad()
            with torch.no_grad():
                pre_classify = model(image)
                _loss = loss_func(pre_classify, classify.view(-1))
                batch_loss = _loss * batch_count
                # calc
                batch_compare_result = pre_classify.argmax(dim=1, keepdim=True).view(-1) == classify
                batch_correct_count = torch.sum(batch_compare_result)
                tested_count += batch_count
                correct_count += batch_correct_count
                batch_correct_acc = 1.0 * batch_correct_count / batch_count
                loss += batch_loss
                if idx % config.log_interval == 0:
                    log(epoch, idx, tested_count, count, batch_correct_acc, correct_count, batch_loss, loss)
