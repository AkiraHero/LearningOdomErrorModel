# **LearningLocErrormodel: Source code for scene aware error modeling of LiDAR/Visual odometry**

This is the source code for the conference paper

Ju, Xiaoliang, et al. "[Learning Scene Adaptive Covariance Error Model of LiDAR Scan Matching for Fusion Based Localization](https://ieeexplore.ieee.org/abstract/document/8813840/)." *2019 IEEE Intelligent Vehicles Symposium (IV)*. IEEE, 2019.

and the submitted journal paper 

Ju,Xiaoliang, et al. "Scene Aware Error Modeling of LiDAR/Visual Odometry for Fusion-based Vehicle Localization", submitted to *IEEE Transactions on Intelligent Transportation Systems*. IEEE, 2020.

## Dependencies

python >3.5

torch >1.0.3

configparser

argparse

matplotlib

pandas

## Usage:

Train the model:

```
python3.5 Main/TrainLocErrorLearningModel.py training-configfile.ini [other options]
```

Test the model:

1) Get raw output of network

```
python3.5 Main/Test-ModelPrediction.py testing-configfile.ini [other options]
```

2) Compare the positioning accuracy using the learned model

```
python3.5 Main/Test-CompareLocAccuracy.py testing-configfile.ini [other options]
```

You can also refer to the Script fold for usage examples.
