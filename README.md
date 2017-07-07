# DeepLab-ResNet-Pytorch


This is an (re-)implementation of [DeepLab-ResNet](http://liangchiehchen.com/projects/DeepLabv2_resnet.html) in TensorFlow for semantic image segmentation on the [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/).

## Updates

**29 Jan, 2017**:
* Fixed the implementation of the batch normalisation layer: it now supports both the training and inference steps. If the flag `--is-training` is provided, the running means and variances will be updated; otherwise, they will be kept intact. The `.ckpt` files have been updated accordingly - to download please refer to the new link provided below.
* Image summaries during the training process can now be seen using TensorBoard.
* Fixed the evaluation procedure: the 'void' label (<code>255</code>) is now correctly ignored. As a result, the performance score on the validation set has increased to <code>80.1%</code>.

**11 Feb, 2017**:
* The training script `train.py` has been re-written following the original optimisation setup: SGD with momentum, weight decay, learning rate with polynomial decay, different learning rates for different layers, ignoring the 'void' label (<code>255</code>).
* The training script with multi-scale inputs `train_msc.py` has been added: the input is resized to <code>0.5</code> and <code>0.75</code> of the original resolution, and <code>4</code> losses are aggregated: loss on the original resolution, on the <code>0.75</code> resolution, on the <code>0.5</code> resolution, and loss on the all fused outputs.
* Evaluation of a single-scale converted pre-trained model on the PASCAL VOC validation dataset (using ['SegmentationClassAug'](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)) leads to <code>86.9%</code> mIoU. This is confirmed by [the official PASCAL VOC server](http://host.robots.ox.ac.uk/anonymous/FIQPRH.html). The score on the test dataset is [<code>75.8%</code>](http://host.robots.ox.ac.uk/anonymous/EPBIGU.html).

**22 Feb, 2017**:
* The training script with multi-scale inputs `train_msc.py` now supports gradients accumulation: the relevant parameter `--grad-update-every` effectively mimics the behaviour of `iter_size` of Caffe. This allows to use batches of bigger sizes with less GPU memory being consumed. (Thanks to @arslan-chaudhry for this contribution!)
* The random mirror and random crop options have been added. (Again big thanks to @arslan-chaudhry !)

**23 Apr, 2017**:
* TensorFlow 1.1.0 is now supported.
* Three new flags `--num-classes`, `--ignore-label` and `--not-restore-last` are added to ease the usability of the scripts on new datasets. Check out [these instructions](https://github.com/DrSleep/tensorflow-deeplab-resnet#using-your-dataset) on how to set up the training process on your dataset.

## Model Description

The DeepLab-ResNet is built on a fully convolutional variant of [ResNet-101](https://github.com/KaimingHe/deep-residual-networks) with [atrous (dilated) convolutions](https://github.com/fyu/dilation), atrous spatial pyramid pooling, and multi-scale inputs (not implemented here).

The model is trained on a mini-batch of images and corresponding ground truth masks with the softmax classifier at the top. During training, the masks are downsampled to match the size of the output from the network; during inference, to acquire the output of the same size as the input, bilinear upsampling is applied. The final segmentation mask is computed using argmax over the logits.
Optionally, a fully-connected probabilistic graphical model, namely, CRF, can be applied to refine the final predictions.
On the test set of PASCAL VOC, the model achieves <code>79.7%</code> of mean intersection-over-union.

For more details on the underlying model please refer to the following paper:


    @article{CP2016Deeplab,
      title={DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs},
      author={Liang-Chieh Chen and George Papandreou and Iasonas Kokkinos and Kevin Murphy and Alan L Yuille},
      journal={arXiv:1606.00915},
      year={2016}
    }


