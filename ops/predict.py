from os.path import basename, dirname


def PREDICT(model, device, loader, count):
    model.eval()
    for _, item in enumerate(loader):
        image = item['image'].to(device)
        classify = model(image)
        result = classify.argmax(dim=1, keepdim=True).item()
        image_path = item["path"][0]
        base_classify = int(basename(dirname(image_path)))
        print(f'image : {image_path}, \tpredict : {result},\tcorrect : {base_classify == result}')
