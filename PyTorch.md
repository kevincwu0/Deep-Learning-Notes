# PyTorch: An Overview

### Introduction
1. PyTorch is a popular Deep Learning framework
- Inspired by the popular Torch deep learning framework written in Lua
- Lua is a big barrier to entry and doesn't offer the modularity like more accessible languages
- AI Researchers, Torch's programming style -> PyTorch but included some key features

2. Imperative Programming (PyTorch)
- Imperative program performs computation as you type it 
- Most python code is imperative 
- runs the actual computation then and there

3. Symbolic Programming
- Clear seperation between defining computation graph
- C= B *A (no computation occurs, symbolic graph is generated -> convert into a function via the compile step)
- Computation occurs as last step of the code

4. Symbolic vs. Imperative
- Tradeoffs
- Symbolic programs are more effective since you can safely resuse the memory of your value for in-place computation
- TensorFlow - Symbolic
- Imperative programs are more flexible (python) -> injecting loops into computation

5. Dynamic Computation Graphs (PyTorch) vs. Static Computation Graphs 
- Define by run -> runtime graph structure
- TensorFlow define and run -> in graph structure writing before compiling, limiting -> assembles graph
- TF computationally expensive, graphs
- Static graphs work well for fixed-size (CNNs) 
- Recurrent Neural Network -> better (tf.while_loop) -> control flow statement
- Dynamic - built and rebuilt -> standard variables
- Any time the work needed is variable -> Dynamic graphs are useful
- Debugging is easy Dynamic
- import toch, from torch.autograd import Variable, FloatTensor, weights, mm, clamp (min and max), SSE loss function, backprop, gradients buffer
- loss.backward(), update weight via gradient descent

6. Summary
- PyTorch uses dynamic computation graphs and imperative programming. 
- Dynamic Graphs are built and rebuilt as necessary at runtime allowing us to use standard python statements
- Imperative programs performs computations as you run them -> no distinction between defining the computation graph and compiling for imperative programs.
- TensorFlow still has the best documentation for a ML library for beginners, for production use since it's built with distributed computing in mind
- PyTorch - for researchers, clear advantage over tensorflow, a lot of new ideas rely on dynamic graphs
