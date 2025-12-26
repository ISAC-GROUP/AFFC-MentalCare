# MentalCare

This is the PyTorch implementation of ***MentalCare*** for psychiatric disorders detection. 

A prior conference peper is accepted as a Spotlight/Interactive paper in the _24th International Conference on Pervasive Computing and Communications (IEEE PerCom 2026)_.
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

Participant samples can be found under the `/Algorithms/data/` directory. We provide physiological data examples for one subject in each group (each psychiatric condition and the healthy control group). The mapping between subject IDs, diagnostic categories, and labels is summarized below:

| Subject IDs              | Category   | Label |
|--------------------------|------------|:-----:|
| Sub18                    | Depression |  0    |
| Sub21                    | Anxiety    |  1    |
| Sub26                    | Bipolar    |  2    |
| Sub3                     | Healthy    |  3    |

## SOTA Baseline Repository 

| Baselines | Venues              | Github Link                                                |
|-----------|---------------------|------------------------------------------------------------|
| ManyDG    | ICLR 2023           | https://github.com/ycq091044/ManyDG                        |
| TSception | IEEE TAFFC 2023     | https://github.com/yi-ding-cs/TSception                    |
| MSTGCN    | IEEE TNSRE 2021     | https://github.com/ziyujia/MSTGCN                          |
| BioDG     | IEEE TETCI 2023     | https://github.com/aristotelisballas/biodg                 |
| CDDG      | Neural Networks 2024| https://github.com/ShaneSpace/DGFDBenchmark                |
| MDNet     | IJCAI 2024          | https://github.com/hairongChenDavid/MDNet                  |



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

## Prototype
As for the hardware design, we provide the schematic diagram of the core board under the `/Prototype/` directory.


## Acknowledgement ✉️

If you are interested in our ubiquitous wearable devices, please contact  [Prof. Yongpan Zou](https://yongpanzou.github.io/), College of Computer Science and Software Engineering, Shenzhen University.
