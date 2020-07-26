"""
Transformations that are applied identically to both the image and the target.
"""
import random
from typing import List, Union, Tuple

import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional as F


class Compose(torchvision.transforms.Compose):
    """
    A modified version of the torchvision Compose transformation, that expects all
    provided transformations to accept an image and a target.
    """

    def __call__(self, image, target):
        for transform in self.transforms:
            result = transform(image, target)
            image, target = result
        return image, target


class RandomGrayscale(torchvision.transforms.RandomGrayscale):
    """
    A modified version of the torchvision RandomGrayscale transformation.
    Does not modify the target.
    """

    def __call__(self, image, target):
        grayscale_image = super().__call__(image)
        return grayscale_image, target


class RandomHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):
    """
    A modified version of the torchvision RandomHorizontalFlip that accepts a target.
    """

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        flipped_image = F.hflip(image)
        width, _ = flipped_image.size
        bbox = target["boxes"]
        bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
        target["boxes"] = bbox
        return flipped_image, target


class RandomVerticalFlip(torchvision.transforms.RandomVerticalFlip):
    """
    A modified version of the torchvision RandomVerticalFlip that accepts a target.
    """

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target
        flipped_image = F.vflip(image)
        _, height = image.size
        bbox = target["boxes"]
        bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
        target["boxes"] = bbox
        return flipped_image, target


class ColorJitter(torchvision.transforms.ColorJitter):
    """
    A modified version of the torchvision ColorJitter that accepts a target.
    Does not modify the target.
    """

    def __call__(self, image, target):
        jittered_image = super().__call__(image)
        return jittered_image, target


class ToTensor(torchvision.transforms.ToTensor):
    """
    A modified version of torchvision ToTensor that accepts a target.
    """

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def _resize_box(new_img_size, img_size, box):
    height, width = img_size
    boxtop, boxleft, boxbot, boxright = box

    if isinstance(new_img_size, int):
        if (width <= height and width == new_img_size) or (
            height <= width and height == new_img_size
        ):
            pass
        if width < height:
            ow = new_img_size
            oh = int(new_img_size * height / width)
            boxtop = boxtop * (oh / height)
            boxbot = boxbot * (oh / height)
            boxleft = boxleft * (ow / width)
            boxright = boxright * (ow / width)
        else:
            oh = new_img_size
            ow = int(new_img_size * width / height)
            boxtop = boxtop * (oh / height)
            boxbot = boxbot * (oh / height)
            boxleft = boxleft * (ow / width)
            boxright = boxright * (ow / width)
    else:
        boxtop = boxtop * (new_img_size[0] / height)
        boxbot = boxbot * (new_img_size[0] / height)
        boxleft = boxleft * (new_img_size[1] / width)
        boxright = boxright * (new_img_size[1] / width)

    return [int(boxtop), int(boxleft), int(boxbot), int(boxright)]


class RandomResizedCrop(torchvision.transforms.RandomResizedCrop):
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

    def bbox_augmentations(
        self,
        img,
        bboxes: List[List[int]],
        labels: List[int],
        size=(256, 256),
        scale=(0.08, 1.0),
        ratio=(0.75, 4 / 3),
        interpolation=Image.BILINEAR,
    ):
        """
        Arguments:
            img: PIL Image
            bboxes: list of bounding boxes [[top, left, bot, right], ...]
            size: image size to convert to
            scale: range of size of the origin size cropped
            ratio: range of aspect ratio of the origin aspect ratio cropped
        """
        top, left, bot, right = torchvision.transforms.RandomResizedCrop.get_params(
            img, scale, ratio
        )
        img = F.resized_crop(
            img, top, left, bot, right, size=size, interpolation=interpolation
        )

        final_boxlist = []
        final_labels = []
        # Assumes box list is [[left, top, right, bot], ...]
        for box, label in zip(bboxes, labels):
            boxleft, boxtop, boxright, boxbot = box

            # remove out-of-frame boxes
            if (
                (left >= boxright)
                or (top >= boxbot)
                or ((top + bot + 1) <= boxtop)
                or ((left + right + 1) <= boxleft)
            ):
                continue

            # cropping
            if top > boxtop:
                boxtop = 0
            else:
                boxtop -= top
            if left > boxleft:
                boxleft = 0
            else:
                boxleft -= left
            if (top + bot) <= boxbot:
                boxbot = bot
            else:
                boxbot -= top
            if (left + right) <= boxright:
                boxright = right
            else:
                boxright -= left

            # resizing
            # to match the same behavior of functional.resize
            boxtop, boxleft, boxbot, boxright = _resize_box(
                size, (bot, right), (boxtop, boxleft, boxbot, boxright)
            )

            # check if less than zero area
            if ((boxbot - boxtop) * (boxright - boxleft)) <= 0:
                continue

            # Point ordering should match https://pytorch.org/docs/stable/_modules/torchvision/models/detection/faster_rcnn.html#fasterrcnn_resnet50_fpn
            final_boxlist.append([boxleft, boxtop, boxright, boxbot])
            final_labels.append(label)

        return img, final_boxlist, final_labels

    def __call__(self, img, target):
        bboxes = target["boxes"]
        labels = target["labels"]
        (
            resized_cropped_img,
            remaining_bboxes,
            remaining_labels,
        ) = self.bbox_augmentations(
            img, bboxes, labels, self.size, self.scale, self.ratio, self.interpolation
        )
        boxes_tensor = torch.Tensor(remaining_bboxes)
        if len(boxes_tensor) == 0:
            # We cropped out all the boxes. Oops. Return empty target
            return (
                resized_cropped_img,
                self.construct_negative_example(target["image_id"]),
            )
        target["boxes"] = torch.Tensor(remaining_bboxes)
        target["labels"] = torch.tensor(remaining_labels, dtype=torch.int64)
        return resized_cropped_img, target
