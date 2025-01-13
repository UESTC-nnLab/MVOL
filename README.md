# Mining In-distribution Attributes in Outliers for Out-of-distribution Detection
**Abstract**:Out-of-distribution (OOD) detection is indispensable for deploying reliable machine learning systems in real-world scenarios. Recent works, using auxiliary outliers in training,
have shown good potential. However, they seldom concern
the intrinsic correlations between in-distribution (ID) and
OOD data. In this work, we discover an obvious correlation that OOD data usually possesses significant ID attributes. These attributes should be factored into the training process, rather than blindly suppressed as in previous approaches. Based on this insight, we propose a structured multi-view-based out-of-distribution detection learning
framework, MVOL, which facilitates rational handling of the
intrinsic in-distribution attributes in outliers. We provide theoretical insights on the effectiveness of MVOL for OOD
detection. Extensive experiments demonstrate the superiority of our framework to others. MVOL effectively handles
both auxiliary OOD datasets and even wild datasets with indistribution data as noise.

This project is for the paper: [Mining In-distribution Attributes in Outliers for Out-of-distribution Detection](https://arxiv.org/abs/2412.11466) published at AAAI 2025. Some parts of the codebase are adapted from [Outlier Exposure](https://github.com/hendrycks/outlier-exposure), and [NTOM](https://github.com/jfc43/informative-outlier-mining).

## Required Packages

The following packages are required to be installed:

- [PyTorch](https://pytorch.org/)
- [Scipy](https://github.com/scipy/scipy)
- [Numpy](http://www.numpy.org/)
- [Sklearn](https://scikit-learn.org/stable/)

Our experiments are conducted with Python 3.8 on NVIDIA 2080 GPUs.

## In-distribution and Auxiliary Outlier Datasets

- In-distribution training set:
  - [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html): included in PyTorch.

* Auxiliary outlier training set:

  * [**300K Random Images**](https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy):provided by [Outlier Exposure](https://github.com/hendrycks/outlier-exposure).
  * [**Mixed CIFAR100**]:auxiliary datasets used in wild settings included in this repository(datasets/wild_datasets).



## Out-of-distribution Test Datasets

We provide links and instructions to download each dataset:

* [SVHN](http://ufldl.stanford.edu/housenumbers/test_32x32.mat): download it and place it in the folder of `datasets/test_ood_datasets/svhn`. Then run `python select_svhn_data.py` to generate test subset.
* [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz): download it and place it in the folder of `datasets/test_ood_datasets/dtd`.
* [Places365](http://data.csail.mit.edu/places/places365/test_256.tar): download it and place it in the folder of `datasets/test_ood_datasets/places365/test_subset`. We randomly sample 10,000 images from the original test dataset.
* [LSUN](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz): download it and place it in the folder of `datasets/test_ood_datasets/LSUN`.
* [LSUN-resize](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz): download it and place it in the folder of `datasets/test_ood_datasets/LSUN_resize`.
* [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz): download it and place it in the folder of `datasets/test_ood_datasets/iSUN`.

## Experiments
Each experiment is conducted in both Single Model Setting and Ensemble Distillation Model Setting with 5 independently runs. Reported results in paper are average of them. Note Single Model Setting is usually adopted in literature. 
### on CIFAR-10
> sh scripts/run_cifar10.sh

### on CIFAR-100
> sh scripts/run_cifar100.sh

### on Wild Settings
> sh scripts/run_wild.sh

## Citation
If you find this work helps your research, please consider citing our paper:

@misc{lei2024miningindistributionattributesoutliers,\
      title={Mining In-distribution Attributes in Outliers for Out-of-distribution Detection}, \
      author={Yutian Lei and Luping Ji and Pei Liu},\
      year={2024},\
      eprint={2412.11466},\
      archivePrefix={arXiv},\
      primaryClass={cs.LG},\
      url={https://arxiv.org/abs/2412.11466}, 
}

