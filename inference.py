import torch
import torch.optim as optim
from model import YOLOv1
from utils import load_checkpoint

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0

model = YOLOv1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
load_checkpoint('overfit.pth.tar', model, optimizer)