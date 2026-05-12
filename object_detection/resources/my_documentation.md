## &#x09;		**Traffic Sign Detection**

#### Preface:

1. This file is a non-formal representation of my own view on this project. Therefore here you will find my thoughts and my own understanding of the implementation only. In future I might write a proper documentation for this project, but it is what it is.
2. Main part of the model is made with the help of Visual Transformers ("AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"), based on object detection technic YOLO ("You Only Look Once: Unified, Real-Time Object Detection")
3. Base dataset for both training and testing time is Tencent100k
4. The idea of this project, as well is the implementation, is made by Alexey Korolyuk.





#### Content:

1. Data Preparation
2. Model Architecture
3. Model Training





##### 1\) Data preparation

We have 3 folders (train, test, other) with photos and a JSON file with annotations (annotations\_all.json). In order to create Dataset class, we need to match photos with the target vector. Let's discuss expected target vector first:

after CNN layers we'll have a vector with shape=(N, N, n\_channels). Each pixel of our new photo representation represents a part of a picture with the help of (n\_channels) values. We will map each pixel with a vector of 5 values (probability, x\_center, y\_center, width, height). Therefore, after flattening, we'll have a vector of length (N \* N \* 5) - labels.

One more thing to notice is that all of the pictures have the same resolution of (2048, 2048). That is the reason for tiling. The plan is to split every image on 4 squares. With the help of that we will get 4 times more images with some parts that has no traffic signs on them (it helps to improve probability parameter), but at the same time we can not afford to many of such images. We will keep only 30% of them. We also need to recalculate coordinate of bounding box in order to correspond to YOLO's specifics (x\_min, x\_max, y\_min, y\_max coordinates need to be recalculated for each tile, but we are saving only relative coordinates). After we get this partition, map them with corresponding recalculated labels, we will get 4 \* (Tencent100k) dataset. Data augmentation is already accomplished on the original dataset + we will add a little of variation to a color jittering. Therefore we have create a processor class for raw data, that saves all images in a correct way with corresponding recalculated labels in a new json file. Now we can get rid of raw data as we have processed images. We also now have a dataset class, that delivers data to a data loader object 'on the fly' and performs color jittering.



##### 2\) Model Architecture

In this section I will explain my thoughts on model architecture. The main idea is to recreate the solution to Traffic Sign Detection task with the help of modern Visual Transformers and compare this two models. Therefore I propose to mix CNN layers with Visual Transformer blocks. Now I'll explain briefly my main idea and later there will be a full diagram with complete description.

On the first step I want to use Convolutional Neural Networks to extract low-level features. We will not only extract features, but also reduce the size of the image to an acceptable size, so the Visual Transformer part can work normally. CNN part of the project is going to be very close to an original paper (1), but less complex, Visual Transformer is also going to be similar to an original paper (3). I will also add a regression layer at the end. So let's conclude: at the beginning we have a picture of shape (3, 640, 640); than we have a couple of CNN layers with 3 \* 3 filters, ReLU activation function and Max Pooling and Linear projection at the end; we now have a representations of shape approximately (20, 20, 256); after that we'll add positional encoding to our representation vector; than we have the most important part - 6-8 blocks of Visual Transformer; at the exit of ViT we have a picture representation of a shape (400, 256) - flattened CNN output vector; the last layer is the main head - a linear layer which gives us 5 values for each target (p, x, y, w, h) with a sigmoid activation function. Almost on every step we will have a dropout regularization technique. The whole model is gonna be created with the bricks (subclasses) such as CNN, MultiheadAttention, Transformer etc.



##### 3\) Model Training

In this section I'll show objects and hyperparameters. For the CNN part we have:

1:

5 Layers. Each layer consists of:

&#x09;Conv2d(kernel\_size=3, stride=1, padding=1)

&#x09;BatchNorm2d()

&#x09;ReLU()

&#x09;MaxPool2d(kernel\_size=2, stride=2)

&#x09;Dropout2d(p=0.25)

Dimensionality:

&#x09;(3, 640, 640) -> (32, 320, 320) -> (64, 160, 160) -> (128, 80, 80) -> (256, 40, 40) -> (512, 20, 20)

2: 

Projection layer

Dimensionality:

&#x09;(20, 20, 512) -> (20, 20, 256)

3:

Parameter embedding layer

4:

8 Transformer layers. Each consists of:

&#x09;LayerNorm(normalized\_shape=256)

&#x09;MultiheadAttention(num\_heads=8, embed\_dim=256)

&#x09;Residual

&#x09;LayerNorm(normalized\_shape=256)

&#x09;FeedForward(embed\_dim=256, expansion=4)

5:

Head

&#x09;Linear(in\_features=256, out\_features=5)

After that we have a (400, 5) vector of predictions. After .view(20, 20, 5) we'll achieve the data at the same format as the target labels. That data can be interpreted properly in the visualization part.



#### References:

1. https://cg.cs.tsinghua.edu.cn/traffic-sign/
2. https://arxiv.org/abs/1506.02640
3. https://arxiv.org/abs/2010.11929

