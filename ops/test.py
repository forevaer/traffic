import torch
from config import config
from utils.log import log
from utils.image import draw_func


def TEST(model, device, loader, count):
    model.eval()
    draw = draw_func() if config.draw else None
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
                batch_acc = 1.0 * batch_correct_count / batch_count
                avg_acc = 1.0 * correct_count / tested_count
                avg_loss = 1.0 * loss / (idx + 1)
                loss += batch_loss
                if draw is not None:
                    draw(batch_loss, avg_loss, batch_acc, avg_acc, epoch == (config.test_epoch - 1))
                if idx % config.log_interval == 0:
                    log(epoch, idx, tested_count, count, batch_acc, avg_acc, batch_loss, avg_loss)
