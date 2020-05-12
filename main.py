# %%
import matplotlib.patches as patches
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import transforms as T
import utils

# %%
from engine import evaluate, train_one_epoch


def get_detection_model(classes: int):
    # load a model pre-trained on COCO
    faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = faster_rcnn.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    faster_rcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, classes)
    return faster_rcnn


# %%
# use our dataset and defined transformations
dataset = CocoDetection(
    "data/images",
    "data/annotations/instances_default.json",
    transforms=T.Compose(
        [
            T.ToFRCNNFormat(),
            T.RandomGrayscale(),
            T.RandomHorizontalFlip(),
            # T.RandomVerticalFlip(),
            T.ToTensor(),
        ]
    ),
)

# split the dataset in train and test set
torch.manual_seed(1)
lengths = [len(dataset) - 50, 50]
dataset_train, dataset_test = torch.utils.data.random_split(dataset, lengths)

# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn,
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn,
)

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# our dataset has two classes only - background and person
num_classes = len(dataset.coco.cats.keys())

# get the model using our helper function
model = get_detection_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# %%
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

# %%
img = Image.open("-22f2Dp1Spo.003.jpg")
to_tensor = T.ToTensor()
img_tensor, _ = to_tensor(img, {})
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img_tensor.to(device)])


im = Image.fromarray(img_tensor.mul(255).permute(1, 2, 0).byte().numpy())

# %%
# Visualize bounding boxes
fig, ax = plt.subplots(1, figsize=(12, 8))
ax.imshow(im)
for box, label, score in zip(
    prediction[0]["boxes"], prediction[0]["labels"], prediction[0]["scores"].round()
):
    if not score.item():
        continue
    [min_x, min_y, max_x, max_y] = box.tolist()
    width = max_x - min_x
    height = max_y - min_y
    rect = patches.Rectangle(
        (min_x, min_y), width, height, linewidth=1, edgecolor="r", facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(min_x, min_y, dataset.coco.cats[label.item()]["name"], c="r")

# %%
torch.save(model, "detector.pt")
