from PIL import Image
import matplotlib.pyplot as plt
from config import config


def load_image(image_path):
    return Image.open(image_path).convert('RGB')


def draw_func(draw_count=config.draw_count):
    batch_loss = []
    loss = []
    batch_acc = []
    acc = []
    plt.ion()

    def draw(_batch_loss, _loss, _batch_acc, _acc, keep=False):
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
        plt.subplot(121)
        plt.title('loss')
        plt.plot(index, draw_loss, c='red', label='loss:{:5.2f}'.format(_loss))
        plt.plot(index, draw_batch_loss, linestyle=':', c='green', label='batch_loss:{:5.2f}'.format(_batch_loss))
        plt.legend(loc='lower right')

        plt.subplot(122)
        plt.title('acc')
        plt.plot(index, draw_acc, c='red', label='acc:{:5.2f}'.format(_acc))
        plt.plot(index, draw_batch_acc, linestyle=':', c='green', label='batch_acc:{:5.2f}'.format(_batch_acc))
        plt.legend(loc='lower right')
        plt.pause(0.1)
        if not keep:
            plt.ioff()
        else:
            plt.show()

    return draw


if __name__ == '__main__':
    a = draw_func()
    for i in range(100):
        a(i, 100 - i, 50 - i, i - 50, i == 99)
