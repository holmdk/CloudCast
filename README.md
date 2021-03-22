# CloudCast
Repo for the [CloudCast: A Satellite-Based Dataset and Baseline for Forecasting Clouds paper](https://ieeexplore.ieee.org/document/9366908)

## Training
```python
python run.py --train True --batch_size 2 --num_gpu 2 --data_backend ddp
```

## Testing
```python
python run.py --train False --batch_size 2 --num_gpu 1 --data_backend dp --pretrained_path './models/pretrained.ckpt'
```

## Visualization
Look into the `saved_images` folder. 

![](./CloudCast/saved_images/batch_0/Output_2018-07-01T07:30_frame_14.png)


## Citation
If you use this work in your own research, please cite the following:

```
@ARTICLE{CloudCastNielsen,
  author={A. H. {Nielsen} and A. {Iosifidis} and H. {Karstoft}},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={CloudCast: A Satellite-Based Dataset and Baseline for Forecasting Clouds}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/JSTARS.2021.3062936}}
```
