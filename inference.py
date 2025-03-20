import torch
import torch.optim as optim
import cv2
from model import YOLOv1
from utils import load_checkpoint, cellboxes_to_boxes, non_max_suppression
import torchvision.transforms as transforms

def draw_boxes(image, boxes, class_labels, color=(0, 255, 0), box_format="midpoint"):
    h, w, _ = image.shape  # Get image dimensions
    
    for box in boxes:
        print('suka', box)
        if len(box) < 6:  # Skip invalid boxes
            continue
        
        class_idx, conf, x, y, width, height = box
        class_idx = int(class_idx)
        
        # Convert from midpoint to corner format
        if box_format == "midpoint":
            x1 = int((x - width / 2) * w)
            y1 = int((y - height / 2) * h)
            x2 = int((x + width / 2) * w)
            y2 = int((y + height / 2) * h)
        else:  # "corners"
            x1, y1, x2, y2 = int(x * w), int(y * h), int(width * w), int(height * h)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label with class name
        label = f"{class_labels[class_idx]} {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0

# transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

model = YOLOv1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
# load_checkpoint('overfit.pth.tar', model, optimizer)
LOAD_MODEL_FILE = "overfit.pth.tar"

image_path = 'data/train/apple_1.jpg'
frame = cv2.imread(image_path)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

input_tensor = transform(frame)
input_tensor = input_tensor.unsqueeze(0)
print('in', input_tensor.shape)

load_checkpoint(torch.load(LOAD_MODEL_FILE, map_location=torch.device(DEVICE)), model, optimizer)
prediction = model(input_tensor)

bboxes = cellboxes_to_boxes(prediction)

iou_threshold=0.5
threshold=0.4
box_format="midpoint"

nms_boxes = non_max_suppression(
    bboxes[0],
    iou_threshold=iou_threshold,
    threshold=threshold,
    box_format=box_format,
)

class_labels = ["apple", "banana", "orange"]  # Example

# Draw bounding boxes on the image
image_with_boxes = draw_boxes(frame, nms_boxes, class_labels)

# all_pred_boxes = []
# for nms_box in nms_boxes:
#     all_pred_boxes.append(nms_box)
cv2.imshow("Detection", image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('a', nms_boxes)