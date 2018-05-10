MNIST_CNN_batch_norm
===
Classification of MNIST_data using CNN or batch normalized cnn

Usage
---
### Command
`python main.py cnn_normal` or `python main.py cnn_batch_norm`

### Arguments
Required
* `select_net` : select cnn_norm or cnn_batch_norm

Optional
* `--activation_func` : choice(sigmoid, relu, lrelu). Default : `relu`
* `--feature_map1` : Number of feature map1. Default : `64`
* `--feature_map2` : Number of feature map2. Default : `128`
* `--feature_map3` : Number of feature map3. Default : `256`
* `--filter_size` : size of filter. Default : `3`
* `--pool_size` : size of max_pooling. Default : `2`
* `--epoch` : Number of epochs to run. Default : `10`
* `--batch_size` : Number of batch_size to run. Default : `100`
* `--learning_rate` : Learning rate for Adam optimizer. Default : `0.001`
* `--drop_rate` : Prob of dropout. Default : `0.7`

Results
---
`python main.py cnn_batch_norm`


![result](/image/cnn_batch_norm(epoch30).PNG)

Reference Implementations
---
+ https://github.com/sjchoi86/Tensorflow-101