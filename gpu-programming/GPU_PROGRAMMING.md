# GPU Programming - Parallel Computing

### Introduction to GPU Programming with CUDA

- GPU Programming 
- Add elements of two arrays together
- Used heavily in deep learning, computer graphics, movies, video games, graphical software, series of instruction (CPU)
- CPU is the main chip to tell, given instructions, in a sequence - graphic sequence

- Step 1: Setup Objects
- Step 2: Transform Coordinates
- Step 3: Transform Coordinates into Camera Space
- Matrix operations for transformation, number of transformations exceeded computation

- GPU Thousands of Cores, Nvidia - massive parallel processing, handling tasks simultaneously
- CPU few cores for sequential
- Document can do serially and in parallel
- Nth Fibonacci can't be done in parallel since we need the previous values
- Deep Learning can have millions of parameters, need GPUs 
- Nvidia GPU - Parallel C Code
- Deep Learning with CuDNN - Tensorflow under the hood
- CUDA - kernal program is the core
  - Kernal is a function that can be executed in parallel on the GPU
  - Executed by an array of CUDA threads - all threads run the same code, each thread has an id used to compute memory address to make controlled decisions
  - Can run thousands of these threads and CUDA organizes these threads into a Grid Hierarchy - Thread Blocks
  - Grid is a set of Thread Blocks that can be processed on the device in parallel, each thread block is a set of concurrent threads that can cooperate among themselves and access a shared memory block
  - Programmer's job to specify grid block specification on each kernal call
- CUDA is a tooklkit that lets programmers use Nvidia's GPUs to parallelize their code 
- CUDA extends C++ with its own programming models consisting of threads, blocks, and grids
- CUDA 6+ uses a unified memory architecture which lets us access allocated data from CPU or GPU code
