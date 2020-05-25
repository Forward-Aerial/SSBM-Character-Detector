# %%
import random

import matplotlib.patches as patches
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import dataset
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
transformations = T.Compose([T.RandomHorizontalFlip(), T.ToTensor()])

dataset = dataset.FRCNNFrameDataset(
    "data/images",
    "data/annotations/instances_default.json",
    transforms=transformations,
)

# split the dataset in train and test set
torch.manual_seed(1)
lengths = [round(len(dataset) * 0.8), round(len(dataset) * 0.2)]
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

# get the model using our helper function
model = get_detection_model(len(dataset.coco.cats.keys()))
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
# %%
num_epochs = 7

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device, dataset.coco)

# %%
img_tensor, test_target = dataset_test[random.randint(0, len(dataset_test))]
to_tensor = T.ToTensor()
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img_tensor.to(device)])


im = Image.fromarray(img_tensor.mul(255).permute(1, 2, 0).byte().numpy())


def visualize_bounding_boxes(
    image: Image, boxes: torch.Tensor, labels: torch.Tensor, scores, color="r"
):
    if scores is None:
        scores = torch.ones(labels.shape)
    _, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    for box, label, score in zip(boxes, labels, scores):
        if score.item() < 0.5:
            continue
        [min_x, min_y, max_x, max_y] = box.tolist()
        width = max_x - min_x
        height = max_y - min_y
        rect = patches.Rectangle(
            (min_x, min_y),
            width,
            height,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            min_x,
            min_y,
            f'{dataset.coco.cats[label.item()]["name"]} ({score.item()})',
            c=color,
        )


prediction_boxes = prediction[0]["boxes"]
prediction_labels = prediction[0]["labels"]
prediction_scores = prediction[0]["scores"]
visualize_bounding_boxes(im, prediction_boxes, prediction_labels, prediction_scores)


visualize_bounding_boxes(
    im, test_target["boxes"], test_target["labels"], None, color="g"
)
# %%
torch.save(model, "detector.pt")
