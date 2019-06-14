## A-CNN: *Annularly Convolutional Neural Networks on Point Clouds*
Created by <a href="https://github.com/artemkomarichev" target="_blank">Artem Komarichev</a>, <a href="http://www.cs.wayne.edu/zzhong/" target="_blank">Zichun Zhong</a>, <a href="http://www.cs.wayne.edu/~jinghua/" target="_blank">Jing Hua</a> from Department of Computer Science, Wayne State University.

![teaser image](https://github.com/artemkomarichev/a-cnn/blob/master/pics/teaser.png)

### Introduction

Our paper (<a href="https://arxiv.org/abs/1904.08017" target="_blank">arXiV</a>) proposes a new approach to define and compute convolution directly on 3D point clouds by the proposed annular convolution.

To appear, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), June 2019

### A-CNN usage

We provide the code of A-CNN model that was tested with Tensorflow 1.3.0, CUDA 8.0, and python 3.6 on Ubuntu 16.04. We run all our experiments on a single NVIDIA Titan Xp GPU with 12GB GDDR5X.

* Classification Task

   <a href="https://1drv.ms/u/s!ApbTjxa06z9CgQfKl99yUDHL_wHs">Download</a> *ModelNet-40* dataset first. Point clouds are sampled from meshes with 10K points (XYZ + normals) per shape and provided by PointNet++.
    
  To train a classification A-CNN model on *ModelNet-40* dataset type the following command:

        python train.py

  To evaluate a trained model run the following script:

        python evaluate.py

* Part Segmentation Task
  
    <a href="https://1drv.ms/u/s!ApbTjxa06z9CgQnl-Qm6KI3Ywbe1">Download</a> *ShapeNet-part* dataset first. Each point cloud represented by 2K points (XYZ + normals) and provided by PointNet++.

    To train a part segmentation A-CNN model on *ShapeNet-part* dataset type the following commands:

        cd part_segm
        python train.py

    To evaluate a trained segmentation model run the following script:

        ./evaluate_job.sh

* Semantic Segmentation Task

    Download <a href="http://buildingparser.stanford.edu/dataset.html">*S3DIS*</a> and <a href="https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip">*ScanNet*</a> datasets provided by PointNet/PointNet++. *S3DIS* contains XYZ + RGB information. *ScanNet* only has geometry information (*XYZ* only), no color.

    To estimate normals we used PCL library. The script to estimate normals for ScanNet data could be found here:

        cd scannet/normal_estimation
        ./run.sh

### Citation
If you find our work useful in your research, please cite our work:

    @InProceedings{komarichev2019acnn,
        title={A-CNN: Annularly Convolutional Neural Networks on Point Clouds},
        author={Komarichev, Artem and Zhong, Zichun and Hua, Jing},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        year={2019}
    }