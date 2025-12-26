# MentalCare

This is the PyTorch implementation of ***MentalCare*** for psychiatric disorders detection. 

A prior conference peper is accepted as a Spotlight/Interactive paper in the 24th International Conference on Pervasive Computing and Communications (IEEE PerCom 2026).
In the extended version submitted to IEEE Transactions on Mobile Computing, we will release the source code, prototype, and a subset of the test samples. 

## Dataset 📖

Regarding the dataset, we collected a real-world, ubiquitous, and multimodal physiological signals from Guangdong
Second Provincial General Hospital and Shenzhen University, including 17 with depression, 22 with anxiety, 16 bipolar with bipolar, as well as 18 normal subjects. We collect thier heart rate variability, blood oxygen saturation, galvanic skin response (IR and IRed), and skin temperature signals. 

| Signal type             | Sensor model | Sampling rate | Data type                                   |
| ----------------------- | ------------ | ------------- | ------------------------------------------- |
| Heart rate              | Pulse sensor | 400 Hz        | Raw analog data                             |
| Blood oxygen saturation | Max30102     | 400 Hz        | Red light and infrared light intensity data |
| Galvanic skin response  | Grove GSR    | 200 Hz        | Voltage simulation data                     |
| Skin temperature        | LMT70        | 100 Hz        | Voltage simulation data                     |



## Code 📖

Environment installation:

- `Python` 3.8

```shell
conda create -n mentalcare python=3.8
```

```shell
conda activate mentalcare
```

Before running our code, please install the following packages:

```
os
seaborn
matplotlib
numpy
tqdm
torch
torchvision
scikit-learn
pandas
```



## Citation 🖊️

If you find our work useful, please consider citing our paper:

```
@ARTICLE{zhang2026depguard,
  author={Zhang, Yufei and Jin, Shuo and Kuang, Wenting and Zheng, Yuda and Song, Qifeng and Fan, Changhe and Zou, Yongpan and Leung, Victor C. M. and Wu, Kaishun},
  journal={IEEE Transactions on Mobile Computing}, 
  title={DepGuard: Depression Recognition and Episode Monitoring System With a Ubiquitous Wrist-Worn Device}, 
  year={2026},
  volume={25},
  number={1},
  pages={197-214},
  doi={10.1109/TMC.2025.3591096}
}
```

## Acknowledgement ✉️

If you are interested in our collection of ubiquitous wearable devices, please contact  [Prof. Yongpan Zou](https://yongpanzou.github.io/), College of Computer Science and Software Engineering, Shenzhen University.
