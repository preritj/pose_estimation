## Pose Detection


### Training on your own dataset
* Currently supports COCO, MPI and PoseTrack datasets, however adding custom dataset is fairly easy. 
 

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





