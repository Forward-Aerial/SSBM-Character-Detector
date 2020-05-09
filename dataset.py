import torch
from PIL import Image
from torch.utils import data


class FoxFrameDataset(data.Dataset):
    def __init__(self, tags_path: str, transforms=None, target_transforms=None) -> None:
        self.transforms = transforms
        self.target_transforms = target_transforms
        with open(tags_path, "r") as fp:
            self.examples = fp.read().split("\n")

    def __getitem__(self, index):
        img_path, min_x, min_y, max_x, max_y, _ = (
            self.examples[index].strip().split(",")
        )
        img = Image.open(img_path).convert("RGB")
        num_objs = 1
        boxes = torch.as_tensor(
            [[int(min_x), int(min_y), int(max_x), int(max_y)]], dtype=torch.float32
        )
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.Tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target

    def __len__(self) -> int:
        return len(self.examples)
