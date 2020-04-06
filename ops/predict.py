from os.path import basename, dirname


def PREDICT(model, device, loader, count):
    model.eval()
    total_count = 0
    correct_count = 0
    for _, item in enumerate(loader):
        total_count += 1
        image = item['image'].to(device)
        classify = model(image)
        result = classify.argmax(dim=1, keepdim=True).item()
        image_path = item["path"][0]
        base_classify = int(basename(dirname(image_path)))
        correct = base_classify == result
        print(f'image : {image_path}, \tpredict : {result},\tcorrect : {correct}')
        correct_count += int(correct)
    print(f"acc : {1.0 * correct_count / total_count}")

