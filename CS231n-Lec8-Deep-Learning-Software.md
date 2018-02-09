# Stanford | Deep learning Software 

### CS231n: Convolutional Neural Networks for Visual Recognition Lecture 8

### Introduction
1. Use of different software packages for deep learning, focusing on TensorFlow and PyTorch
2. Discuss some differences between CPUs and GPUs
3. Fancier optimization: SGD + Momentum, Nesterov, RMSProp, Adam (vanilla SGD) 
  - Easier to implement but make your networks converge a bit faster
4. Regularization -> dropout 
  - add noise during training, then marginalize out during testing so not stochastic
5. Transfer Learning 
  - Download big network, fine-tune for own problem, tackle DL
6. Writing software and how hardware works
- CPU vs GPU
- Deep Learning Frameworks
  - Caffe / Caffe2
  - Theano / TensorFlow
  - Torch / PyTorch
7. CPU vs GPU
  - Deep Learning using GPUs
  - Why GPUs > CPU
  - Who built a computer before? 
  - Central Processing Unit (CPU)
  - GPu master thing (cooling, quite large)
  - GPU graphics card - processing unit - rendering graphics (games)
  - NVIDIA vs AMD 
    - NVIDIA -> Deep Learning
    - Large focus, hardware suited for Deep Learning (NVIDIA GPUS)
  - CPU vs GPU
    - CPU - fewer cores, but each core is much more capable, great at sequentials tasks
    - GPU More cores, much more slower and "dumber"; great for parallel tasks -> work together
    - GPU good for parallel programming 
    - CPU small cache -> pulling from RAM
    - GPU own RAM -> bottleneck between computer GPU, Titan Xp, own caching system
  - GPU -> Matrix Multiplication
    - Final matrix - dot product one of the columns row of matrix
    - output matrix in parallel - dot product of vectors, GPU elements computed in parallel
    - CPU sequentially
    - CPU vectorize program -> today
  - Convolution 
    - input tensor, weigt tensor, output tensor (parallel GPUs) 
  - CUDA abstractions (C-Like code) 
    - really tricky, memory heirarchy, performant, lots of library highly optimized
    - cuBLAS(matrix multiplication), cuDNN (convolution)
    - OpenCL usually much slower 
  - Udacity: Intro to Parallel Programming -> Check it out! Paradigm of the code
  - How it works etc.
  - VGG16 65x speed up GPU vs CPU (benchmarks, unfair a lot benchmarks) - maximal performance - substantial
  - optimized cuDNN -> naive open source vs (3x speed up), GPU (cuDNN)
  - Bottleneck, reading of sequntial, read all data into RAM, SSD instead of HHD, use multiple CPU threads
  - Software - prefetching CPUs - minibatch to GPUs
 8. Various Deep Learning Frameworks
  - Caffe, Torch, Theano, TensorFlow
  - Caffe2(Facebook), PyTorch (Facebook), Paddle (Baidu), CNTK (Microsoft), MXNet(Amazon)
  - Built v1 in Academia, v2 industry
  - PyTorch vs TensorFlow
  - Torch -> PyTorch (most experience)
9. Computational Graphs
  - regularization, graph structure, different weights, computational graphs messy
  - Three reasons to use deep learning frameworks
    - 1. Easily build big computational graphs
    - 2. Easily compute gradients in computational graphs
    - 3. Run it all efficiently on GPU (wrap CuDNN, cuBLAS, etc.) 
  - computing loss, gradients + backprop, running efficiently, low-level details 
  - Numpy - computational graphs easy
  - Numpy CPU only - have to compute own gradients
  - Let forward-pass, GPU automatically compute gradients, similar to numpy syntax
  - TensorFlow can switch to CPU -> GPU, .cuda (PyTorch)
10. TensorFlow: Neural Nets
  - Example: Train a two-layer ReLu network on random data with L2 Euclidean loss
  - TensorFlow -> two computation stages (define graph, run graph and reuse many time)
  - TensorFlow.session (run graph)
  - sess.run to actually run -> loss, feed in params
  - weights (define as variable, live in computation graph, and persist, tf.random_normal)
  - funny indirection in TensorFlow, TensorFlow magic, TensorFlow.group
  - optimizer = tf.train.GradientDescentOptimizer(1e-5) -> compute gradients and update weights
  - tf.variable, basic functions for us -> TensorFlow:loss, chain together (params, matrix multi. batch normalizations, combining hard)
  - Xavier initializer -> out an h, activation.tf.relu -> 
11. Keras
  - High-level wrapper for TF
  - model.compile -> model.fit
  - TFLearn, TensorLayer, tf.layers, TF-Slim, tf.contrib.learn, PRetty Tensor, Sonnet (DeepMind)
  - Tensorflow: pre-trained models
  - TensorBoard: plot losses etc.
  - TensorFlow: Distributed Version
  - Theano (earlier framework from Montreal)
12. PyTorch
  - Three layers of abstractions
    - 1. Tensor: Imperative ndarray (like numpy array) but runs on GPU 
    - 2. Variable: Node in a computational graph; stores data and gradient
    - 3. Module: a neural network layer (compose together); may store state or learnable weights
    - Tensor -> Numpy Array, Varialbe -> Tensor, Module -> Sonnet, TFLearn
  - PyTorch Tensors are like numpy arrays, but can run on GPU
  - PyTorch Tensors run on GPU -> use a cuda datatype, cast datatype, Pytorch + GPU
  - Computation graphs, x.grad.data -> a tensor of gradients
  - PyTorch Tensors and Variables have the same API
  - PyTorch - Autograd
  - TF explicit graph and running grap, PyTorch new graph more cleaner, autograd function (by writing forward and backward for tensors)
  - Define own ReLu -> most of the time no need for Autograd
  - PyTorch: nn package
  - Higher-level wrapper 
  - nn looks like keras, each iterations -> prediction to loss function, gradient descent 
  - PyTorch optimizer for different update rules
  - optim 
  - nn Define new Modules - Module neural network layers; it inputs and outputs Variables; Modules can contain weights (as variables) or other Modules)
  - You can define your own modules using autograd!
  - DataLoaders wraps a dataset and provides minibatching, shuffling, multithreading for you
  - When you need to load custom data, just write your own Dataset class
  - Pretrained Models -> vgg16 = torchvision.models.vgg16(pretrained=True), alexnet = torchvision.models.alexnet(pretrained=True)
  - Visdom - similar to TensorBaord, computational graph (TensorBoard)
  - Torch -> written in Lua, same backend C code 
  - PyTorch (+) autograd -> similar to write complex code (Torch doesn't have this)
 13. Static vs. Dynamic Graphs
 - TensorFlow: build graph once, then run many times (static)
 - PyTorch: Each forward pass defines a new graph (dynamic)
 - Feed Forward Neural Networks (no difference)
 - Static vs Dynamic: Optimization
 - Static graphs can optimize the graph for you before you run it! (ReLu) much more efficiently
 - Static graphs -> dynamic graphics
 - Serialization - Data structure in memory -> serialize to disk
 - Dynamic building and execution, Static (serialize and run)
 - Dynamic (nicer super simple) -> similar to numpy
 - TensorFlow -> control flow oeprator (tensorflow baked -> paths once -> baked python control flow) 
 - Loops -> normal python, TensorFlow functional programming tf.fold + control flow, imperatively, any control flow
 - TensorFlow Fold -> implement through, dynamic batching, awkward (not as pytorch)
 
 14. Why care about dynamic graphs?
 - Recurrent networks
 - Recursive Networks - parse tree, not a sequential tree structure
 - Neuromodule Modular Networks (read questions, colors and finding cats, compile custom architecture)
 - Dynamic Computational Graphs (cool ideas)
 
 15. Caffe (UC Berkeley)
 - Written in C++, python binders
 - edit prototxt (some inner product)
 - downside (ugly for large big models) ->  VVG, not documented -> does work
 
 16. Caffe2
 - define computational graph structure in python
 - PyTorch (Research), Caffe2 (Production)
 - TensorFlow (one framework to rule them all)

 17. TensorFlow safe bet for most projects -> higher level wrappers (Keras, Sonnet)
 18. PyTorch is best for research, less code out there
 19. Caffe, Caffe2, or TensorFlow
