from PIL import Image
import matplotlib.pyplot as plt
from config import config
from sklearn.metrics import confusion_matrix
import numpy as np


def load_image(image_path):
    # 使用三通道，对颜色敏感，不利于泛化，采用灰度图像进行推理
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(e)
        raise Exception(f'illegal image_format : {config.image_format}')


def draw_func(draw_count=config.draw_count):
    batch_loss = []
    loss = []
    batch_acc = []
    acc = []
    plt.ion()

    def draw(epochs, epoch, cover, total, _batch_loss, _loss, _batch_acc, _acc, keep=False):
        batch_loss.append(_batch_loss)
        loss.append(_loss)
        batch_acc.append(_batch_acc)
        acc.append(_acc)
        length = len(loss)
        start = 0
        if (draw_count is not None) and (draw_count > 0) and (draw_count < length):
            draw_batch_loss = batch_loss[-draw_count:]
            draw_loss = loss[-draw_count:]
            draw_batch_acc = batch_acc[-draw_count:]
            draw_acc = acc[-draw_count:]
            start = length - draw_count
        else:
            draw_batch_loss = batch_loss
            draw_loss = loss
            draw_batch_acc = batch_acc
            draw_acc = acc

        index = [x for x in range(start, length)]
        plt.clf()
        plt.subplot2grid((1, 2), (0, 0))
        plt.title('loss')
        plt.plot(index, draw_loss, c='red', label='loss:{:5.2f}'.format(_loss))
        plt.plot(index, draw_batch_loss, linestyle=':', c='green', label='batch_loss:{:5.2f}'.format(_batch_loss))
        plt.legend(loc='lower right')
        plt.subplot2grid((1, 2), (0, 1))
        plt.title('acc')
        plt.plot(index, draw_acc, c='red', label='acc:{:5.2f}'.format(_acc))
        plt.plot(index, draw_batch_acc, linestyle=':', c='green', label='batch_acc:{:5.2f}'.format(_batch_acc))
        plt.suptitle(
            f'WEIGHT:{config.weight_switch_on.name} - FORMAT:{config.image_format.name}\nPHASE : {config.phase.name} - EPOCH : {epoch + 1:.0f}/{epochs:.0f} - PROCESS: {cover:.0f}/{total:.0f}',
            color='red', backgroundcolor='yellow')
        plt.legend(loc='lower right')

        plt.tight_layout(rect=(0, 0, 1, 0.9))
        plt.pause(0.1)
        if not keep:
            plt.ioff()
        else:
            plt.show()

    return draw


def draw_confusion(origin, predict):
    classes = sorted(list(set(origin + predict)))
    classes_idx = range(len(classes))
    confusion = confusion_matrix(origin, predict)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    plt.xticks(classes_idx, classes)
    plt.yticks(classes_idx, classes)
    plt.colorbar()
    plt.xlabel('origin')
    plt.ylabel('predict')
    for predict_idx in range(len(confusion)):
        for origin_idx in range(len(confusion[predict_idx])):
            plt.text(origin_idx, predict_idx, confusion[predict_idx][origin_idx])
    plt.show()


if __name__ == '__main__':
    # a = draw_func()
    # for i in range(100):
    #     a(i, 100 - i, 50 - i, i - 50, i == 99)
    draw_confusion([1, 2, 3, 4, 5, 6, 4, 3, 2, 1, 2, 3], [1, 5, 6, 8, 5, 4, 3, 3, 2, 2, 2, 3])
