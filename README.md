# SSBM Character Detector (PyTorch)

![GIF of Captain Falcon and Jigglypuff being tracked](output.gif)

A multi-label object detection model used to identify where character are in a Super Smash Bros. Melee tournament video frame.
Inspired by [Adam Spannbauer's `ssbm_fox_detector`](https://github.com/AdamSpannbauer/ssbm_fox_detector), which focused on just Fox, I want my model to be able to correctly identify all characters in a frame and draw bounding boxes around them.

Your input data will need to be in COCO format.
If you'd like to use the same dataset that I did, please get in contact with me.
Because of size constraints (~13GB), I'm not going to be distributing the dataset with this repository.

If you DON'T have input data, you can download Adam Spannbauer's labeled dataset (though it's only labels of purple Fox), accessible at the link above. To convert his dataset into COCO format, just run `utils/convert_txt_to_coco.py`.

## Dependencies

All listed dependencies are for Ubuntu 20.04

1. Python 3.8
2. CUDA 10.0
3. Everything in `requirements.txt`
4. Also, there's some sort of bug that requires `pycocotools` to be manually installed from a Git repo, so also do `pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`

## Running the Script

Just run `main.py`, which is modified from [this tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#testing-forward-method-optional).
For those who care, it creates a DETR model.
I have no idea what that means, but it sure sounds cool!
