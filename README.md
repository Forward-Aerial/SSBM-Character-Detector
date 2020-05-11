# SSBM Fox Detector (PyTorch)

While working on a bigger project, I stumbled upon [Adam Spannbauer's `ssbm_fox_detector`](https://github.com/AdamSpannbauer/ssbm_fox_detector) repository, which was written 3 years ago for Keras and TensorFlow. 
Despite trying to reverse-engineer dependencies, I couldn't get the scripts to run.
I'm going to attempt to use PyTorch's built-in Faster-RCNN model to try and replicate the results so that I can use this for some other purposes.

## Dependencies
All listed dependencies are for Ubuntu 20.04
1. Python 3.8
2. CUDA 10.0
3. Everything in `requirements.txt`
4. Also, there's some sort of bug that requires `pycocotools` to be manually installed from a Git repo, so also do `pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`

The notebook to look at is `torchvision_finetuning_instance_segmentation.ipynb`, modified from [this tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#testing-forward-method-optional).
For those who care, it creates an F-RCNN model with a ResNet-50 backbone (that uses a Feature Pyramid Network).
