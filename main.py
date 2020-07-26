# %%
import random
import time

import matplotlib.patches as patches
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import dataset
import transforms as T
import utils
from coco_eval import CocoEvaluator
from engine import WeirdLossException, evaluate, train_one_epoch

TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 1

PRINT_FREQUENCY = 50

# %%


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
transformations = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomGrayscale(),
        T.ColorJitter(),
        T.RandomResizedCrop(size=(256, 256)),
        T.ToTensor(),
    ]
)

test_transformations = T.Compose([T.ToTensor(),])

dataset_train = dataset.FRCNNFrameDataset(
    "data/images",
    "data/annotations/instances_default.json",
    transforms=transformations,
)

dataset_test = dataset.FRCNNFrameDataset(
    "data/images",
    "data/annotations/instances_default.json",
    transforms=test_transformations,
)

# split the dataset in train and test set
torch.manual_seed(1)
length = np.arange(len(dataset_train))
# comment out this line if you don't want to permute the training and test set
length = np.random.permutation(length)

split_at = round(len(dataset_train) * 0.8)
datasubset_train = torch.utils.data.Subset(dataset_train, length[:split_at])
datasubset_test = torch.utils.data.Subset(dataset_test, length[split_at:])

# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(
    datasubset_train,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    collate_fn=utils.collate_fn,
)

data_loader_test = torch.utils.data.DataLoader(
    datasubset_test,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    collate_fn=utils.collate_fn,
)

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on {device}")

# get the model using our helper function
model = get_detection_model(len(dataset_train.coco.cats.keys()))
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# %%
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
            f'{dataset_train.coco.cats[label.item()]["name"]} ({score.item()})',
            c=color,
        )


# %%
num_epochs = 5

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    try:
        train_one_epoch(
            model,
            optimizer,
            data_loader_train,
            device,
            epoch,
            print_freq=PRINT_FREQUENCY,
        )
    except WeirdLossException as e:

        for img_tensor, target in zip(e.images, e.targets):
            print(target)
            im = Image.fromarray(
                img_tensor.cpu().mul(255).permute(1, 2, 0).byte().numpy()
            )
            visualize_bounding_boxes(
                im, target["boxes"], target["labels"], None, color="r"
            )
            plt.show()
            plt.waitforbuttonpress()
        raise Exception("Weird loss")

    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    coco_evaluator: CocoEvaluator = evaluate(model, data_loader_test, device)

# %%
idx = random.randint(0, len(dataset_test))
img_tensor, test_target = dataset_test[idx]
# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img_tensor.to(device)])


im = Image.fromarray(img_tensor.mul(255).permute(1, 2, 0).byte().numpy())

prediction_boxes = prediction[0]["boxes"]
prediction_labels = prediction[0]["labels"]
prediction_scores = prediction[0]["scores"]
visualize_bounding_boxes(im, prediction_boxes, prediction_labels, prediction_scores)


visualize_bounding_boxes(
    im, test_target["boxes"], test_target["labels"], None, color="g"
)
# %%
torch.save(model, "detector.pt")

for i in range(5):
    print("\a")
    time.sleep(1)
