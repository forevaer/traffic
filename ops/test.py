import torch
from config import config, enum
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
                # calc
                positive_smaple = pre_classify.argmax(dim=1, keepdim=True).view(-1) == classify
                batch_correct_count = torch.sum(positive_smaple)
                if config.weight_switch_on.value:
                    # 不同预测结果采取不同权值激励，以提高对预测错误的样本的感知<三通道中，对颜色敏感，降低颜色敏感>
                    negative_sample = positive_smaple == False
                    positive_loss = 0
                    negative_loss = 0
                    batch_error_count = torch.sum(negative_sample)
                    if batch_correct_count > 0:
                        positive_loss = loss_func(pre_classify[positive_smaple], classify[positive_smaple].view(
                            -1)) * batch_correct_count * config.positive_weight
                    if batch_error_count > 0:
                        negative_loss = loss_func(pre_classify[negative_sample], classify[negative_sample].view(
                            -1)) * batch_error_count * config.negative_weight
                    batch_loss = positive_loss + negative_loss
                else:
                    batch_loss = loss_func(pre_classify, classify.view(-1)) * batch_count * config.normal_weight
                tested_count += batch_count
                correct_count += batch_correct_count
                batch_acc = 1.0 * batch_correct_count / batch_count
                avg_acc = 1.0 * correct_count / tested_count
                avg_loss = 1.0 * loss / (idx + 1)
                loss += batch_loss
                if draw is not None:
                    draw(config.test_epoch, epoch, tested_count, count, batch_loss, avg_loss,
                         batch_acc, avg_acc, (epoch == (config.test_epoch - 1)) and (tested_count == count))
                if idx % config.log_interval == 0:
                    log(epoch, idx, tested_count, count, batch_acc, avg_acc, batch_loss, avg_loss)
