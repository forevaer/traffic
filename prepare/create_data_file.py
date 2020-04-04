import os
import pandas as pd
from threading import Lock

lock = Lock()


def path_name(*path):
    return os.path.join(path[0], *path[1:])


def save_csv(data_dict, path):
    data = pd.DataFrame(data_dict)
    data.to_csv(path)


def _format_data(root, phase):
    prefix = path_name(root, phase)
    dirs = sorted(os.listdir(prefix))
    path = []
    classify = []
    for _dir in dirs:
        image_dir = path_name(prefix, _dir)
        images = [x for x in os.listdir(image_dir)]
        # 排序操作，避免重名改名失败
        images = sorted(images)
        count = len(images)
        cls = int(_dir)
        for idx in range(count):
            lock.acquire()
            origin_name = images[idx]
            old_name = path_name(image_dir, origin_name)
            fix_name = f'{_dir}_{idx}.png'
            new_name = path_name(image_dir, fix_name)
            os.rename(old_name, new_name)
            classify.append(cls)
            path.append(new_name)
            lock.release()
    save_csv({
        'path': path,
        'classify': classify
    }, f'../{phase}.csv')


def format_data(root, *phase):
    for _phase in phase:
        _format_data(root, _phase)


if __name__ == '__main__':
    format_data("../data", 'train', 'test')
