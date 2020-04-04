from config import config
from config.enum import PHASE
from net.net import Net
from utils.data import prepare_model, get_loader
import ops

device = config.device()
model = Net().to(device)
model = prepare_model(model)
loader, count = get_loader()
if config.phase is PHASE.TRAIN:
    ops.TRAIN(model, device, loader, count)
elif config.phase is PHASE.TEST:
    ops.TEST(model, device, loader, count)
elif config.phase is PHASE.PREDICT:
    ops.PREDICT(model, device, loader, count)
