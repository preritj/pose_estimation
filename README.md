## Pose Estimation
Tensorflow implementation for real-time multi-person pose estimation. Based on [PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model](https://arxiv.org/abs/1803.08225). Model features include

* Heatmaps to predict keypoints location
* Vectormaps which predict vectors at each keypoint location pointing to adjacent keypoints 
* Offsetmaps predict further refined location of each keypoint.


### Training on your own dataset 

* You will have to modify `dataset/data_reader.py` according to your data format. Example for COCO dataset is provided. 
 
* Training and model parameters can be set in `config.py`
 
* For single GPU training, simply run: 
```
python3 train.py
``` 
 
* For distributed training, first install [horovod](https://github.com/uber/horovod). Then run: 
```
mpirun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 train_distributed.py
```

* Code for tf-summary for visualization during training can be found in `train.py`. 


### Pre-trained model
* Pre-trained model is available in `models/latest`. You may have a look at `inference.py` on how to use the frozen model. 
* This model uses Mobilenet backbone for real-time inference.
* Model is fully-convolutional and supports RGB image of any dimensions (multiple of 16).
* Code for post-processing step using greedy algorithm similar to that defined in https://arxiv.org/abs/1803.08225  can be found in `utils/visualize.py`.


### Results 

Raw output: 

![Sample1-raw](extras/out_heatmap.png) 

Post-processed output: 

![Sample1-postprocess](extras/out_instances.png)


Raw output: 

![Sample1-raw](extras/out_heatmap_3.png) 

Post-processed output: 

![Sample1-postprocess](extras/out_instance_3.png)


