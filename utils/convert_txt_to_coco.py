"""
A script that takes Adam Spannbauer's data
(https://drive.google.com/drive/folders/17QcuO9GQsiqO_4V86iV1gBGUCCfbPkUm)
and converts it into the COCO dataset format.
"""
import argparse
import datetime
import json
import os
import shutil
from typing import IO, Any, Dict, List, NamedTuple

from PIL import Image as PILImage

INFO = {
    "description": "SSBM Fox Detection Dataset",
    "url": "https://github.com/Forward-Aerial/ssbm_fox_detector_pytorch",
    "version": "0.1.0",
    "year": 2020,
    "contributor": "SaltyQuetzals",
    "date_created": datetime.datetime.now().isoformat(" "),
}

LICENSES = []

CHARACTER_NAMES = [
    "Bowser",
    "Captain Falcon",
    "Donkey Kong",
    "Dr. Mario",
    "Falco",
    "Fox",
    "Ganondorf",
    "Ice Climbers",
    "Jigglypuff",
    "Kirby",
    "Link",
    "Luigi",
    "Mario",
    "Marth",
    "Mewtwo",
    "Mr. Game & Watch",
    "Ness",
    "Peach",
    "Pichu",
    "Pikachu",
    "Roy",
    "Samus",
    "Sheik",
    "Yoshi",
    "Young Link",
    "Zelda",
    "CHARACTER"
]
CATEGORIES = [
    {"name": character_name, "supercategory": "", "id": i}
    for i, character_name in enumerate(CHARACTER_NAMES)
]

IMAGE_ID_MAP = {}


class SpannbauerEntry(NamedTuple):
    """
    Represents a row in Adam Spannbauer's dataset file, fox_frcnn_tags.txt
    """

    image_path: str
    min_x: int
    min_y: int
    max_x: int
    max_y: int
    category: int


def area(entry: SpannbauerEntry) -> int:
    """
    Calculates the area of a rectangle as defined by the top-left and bottom-right points.
    """
    return (entry.max_x - entry.min_x) * (entry.max_y - entry.min_y)


def process_tag_line(line: str) -> SpannbauerEntry:
    """
    Converts a line in the tags.txt file into a workable data format.
    """
    (
        image_path_str,
        min_x_str,
        min_y_str,
        max_x_str,
        max_y_str,
        category_str,
    ) = line.split(",")
    for character_category in CATEGORIES:
        if category_str.lower() == character_category["name"].lower():
            category = character_category["id"]
    return SpannbauerEntry(
        image_path_str,
        int(min_x_str),
        int(min_y_str),
        int(max_x_str),
        int(max_y_str),
        category,
    )


def process_tags_file(tags_file: IO[Any]) -> List[SpannbauerEntry]:
    lines = tags_file.read().splitlines()
    return list(map(process_tag_line, lines))


def build_annotation(annotation_id: int, entry: SpannbauerEntry) -> Dict:
    """
    Constructs a COCO annotation from an annotation ID and entry data.
    """
    width = entry.max_x - entry.min_x
    height = entry.max_y - entry.min_y
    points = [
        entry.min_x,  # Top left
        entry.min_y,
        entry.min_x + width,  # Top right
        entry.min_y,
        entry.min_x + width,  # Bottom right
        entry.min_y + height,
        entry.min_x,  # Bottom left
        entry.min_y + height,
    ]
    return {
        "iscrowd": 0,
        "image_id": IMAGE_ID_MAP[entry.image_path],
        "bbox": [entry.min_x, entry.min_y, width, height],
        "category_id": entry.category,
        "segmentation": [points],
        "id": annotation_id,
        "area": area(entry),
    }


def main(tags_filepath: str):
    with open(tags_filepath, "r") as tags_file:
        entries = process_tags_file(tags_file)
    images = []
    annotations = []
    response = input(
        "Executing this script will irreversibly modify the provided data. Are you sure you want to continue? [y/N]"
    )
    if not response or response.lower() == "n":
        print("Aborting execution.")
        return
    for i, entry in enumerate(entries):
        if entry.image_path not in IMAGE_ID_MAP:
            new_folder = "data/images"
            if not os.path.exists(new_folder):
                os.mkdir(new_folder)
            new_filename = f"{new_folder}/{i}.jpg"
            os.rename(entry.image_path, new_filename)
            img = PILImage.open(new_filename)
            width, height = img.size
            IMAGE_ID_MAP[entry.image_path] = len(IMAGE_ID_MAP)
            images.append(
                {
                    "date_captured": 0,
                    "flickr_url": 0,
                    "height": height,
                    "width": width,
                    "id": IMAGE_ID_MAP[entry.image_path],
                    "license": 0,
                    "file_name": os.path.basename(new_filename),
                }
            )
        annotations.append(build_annotation(i, entry))
    if not os.path.exists("data/annotations"):
        os.mkdir("data/annotations")
    json.dump(
        {
            "licenses": LICENSES,
            "images": images,
            "annotations": annotations,
            "info": INFO,
            "categories": CATEGORIES,
        },
        open("data/annotations/instances_default.json", "w+"),
    )
    shutil.rmtree("data/tbh7_purp_fox_vgbc",)
    shutil.rmtree("data/wiz_sfat")
    os.remove("data/fox_annotations.xml")
    os.remove("data/image_metadata_stylesheet.xsl")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility script for converting Adam Spannbauer's ssbm_fox_detector data into the COCO dataset format. See https://github.com/AdamSpannbauer/ssbm_fox_detector for links to the data."
    )
    parser.add_argument(
        "tags_filepath", type=str, help="Path to the fox_frcnn_tags.txt file."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the contents of data.zip (see https://drive.google.com/drive/folders/17QcuO9GQsiqO_4V86iV1gBGUCCfbPkUm)",
    )
    args = parser.parse_args()
    labels = [{"name": name} for name in CHARACTER_NAMES]
    main(args.tags_filepath)
