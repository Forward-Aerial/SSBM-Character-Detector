import torch
from PIL import Image
from torchvision import datasets
import os


class FRCNNFrameDataset(datasets.CocoDetection):
    """
    Changes output of CocoDetection to fit FRCNN format requirements.
    """

    def coco_target_to_frcnn(self, coco_target):
        """
        Converts a COCO target into an FRCNN target.
        """
        boxes = []
        labels = []
        if not isinstance(coco_target, list):
            coco_target = [coco_target]
        image_id = torch.tensor([coco_target[0]["image_id"]])
        for annotation in coco_target:
            [min_x, min_y, width, height] = annotation["bbox"]
            max_x = min_x + width
            max_y = min_y + height
            box = [min_x, min_y, max_x, max_y]
            label = annotation["category_id"]

            boxes.append(box)
            labels.append(label)

        boxes_tensor = torch.Tensor(boxes)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        return {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": image_id,
        }

    def construct_negative_example(self, img_id: int):
        """
        Constructs a negative example (where no objects are present).
        """
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        return {
            "boxes": boxes,
            "labels": torch.zeros((1, 1), dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
        }

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]["file_name"]

        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        if len(target) == 0:
            # Negative example with no bounding boxes.
            frcnn_target = self.construct_negative_example(img_id)
        else:
            frcnn_target = self.coco_target_to_frcnn(target)
        if self.transforms is not None:
            img, frcnn_target = self.transforms(img, frcnn_target)
        return (img, frcnn_target)


class FRCNNFrameCharacterDataset(FRCNNFrameDataset):
    """
    Sets all of the labels of examples as "CHARACTER".
    """

    def __getitem__(self, index):
        (img, frcnn_target) = super().__getitem__(index)
        character_label_id = list(self.coco.cats.keys())[-1]
        old_labels_tensor: torch.Tensor = frcnn_target["labels"]
        frcnn_target["labels"] = torch.tensor(
            [character_label_id for _ in range(old_labels_tensor.shape[0])],
            dtype=torch.int64,
        )
        print(frcnn_target)
        return (img, frcnn_target)
