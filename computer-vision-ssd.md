### Single Shot Detection (SSD) Algorithm

### Plan of Attack
1. How Single Shot Detection (SSD) is different
2. Multi-box concept
3. Predicting Object Positions
4. The scale problem

Artificial Neural Networks
Convolutional Neural Networks


### How SSD is different
- Single Shot Multibox detection algorithm is different to others
- Image - sheep walking around, computer or object detection
- Computational tricks and hacks (huge portion of traditional algorithms) would engage in something called an object proposal methodology.
- Object proposal techniques - convert to rectangle, waste of time to find the sheep, object proposal: segment to parts to suggest where there could possibly be objects and not using gradient and colors (colors, no edges, same textures), saves computational time but sacrifice accuracy
- high percentage accuracy, needs to happen in real-time
- SSD came up with a brand new solution
  - Do everything in one-shot, looks at the image once, doesn't haven't to run many convolutional neural network, (YOLO algorithm, you only look once, SSD do better than YOLO), no other algorithm (looks at the image once)
  - Train up on detecting object, same network (reduce the image size of the juices), lots of hacks into the hack
  - most important it happens in one huge convolution
  - 300 x 300 Image
  - VGG-16 through Conv5, 3 -layers
  - Classifer (Extra feature layers)
  - Detections
  - We https://arxiv.org/pdf/1512.02325.pdf
  
### The multi-box concept
- Ground-Truth, SSD identify build this box
- Ground-Truth: seperates empirical evidence from inferred evidence - computer vision, observed evidence
- training networks SSD -> we need ground truth, cannot train the algorithm if these boxes are not present in the first time
- SSD Algorithm
  - Break down into segments
  - Construct several boxes -> ask for every single class of boxes it is training for, none of these have an object of interest,
  - sufficient number of features (80 - 70%), sees enough features would be enough for SSD algorithm (that would be a draw), all these boxes predict something, we'll predict something
  - rather than just detect (labeled vs. a person or not) we need to know exactly where the person is
  - each box as a seperate image, then yes similar to CNN, predict if human inside box
  - seperate images in their own right, aggregate error and propropagate it.
  - error is when comparing to ground truth, error backpropagated to network

### Predicting Object Positions
  - Overlay a grid over image, every single center of grrid is lots of rectangles, each rectange think of each as an image. 
  - CNN -> checks if there's a boat in the after
  - rectangles by itself is there any features of an object, after training in the ideal scenario, aftertraining features of boats, do overlap with ground truth. Iterative training, lengthy hard part, training and application, none of see the full ground truth of the image, where to predict full boat, how do you know where the full part of the boat?
  - We have the ground truth, training (actual boxes), mitigated through training -> box compared to ground truth (how error is calculated, error backpropagated through the network, gradient descent)
  - Takeaways through training:
    - 1) that each box will learn better classify and identify if an object is in there, ground truth to assess it
    - 2) Better at identify final output rectangle, larger and shorter rectangle
  - Training allows it to identify objects through the ground truth but allows it to identify the correct widts and heights of those boxes. 
  
### The Scale Problem
  - What is the Scale Problem?
  - Horses in a field -> apply what we know, grid -> boxes -> features of horses, horse in front (most obvious has been missed) why doesn't see the horse point blank (it's too big for the algorithm to pick up)
  - single out rectangle (works as a seperate image, do I see an image of a horse or not), horse or not, isolate rectangle can't see anything (hooves, nose, and eyes) - SSD horse there, make a suggestion the actual box where the object is
  - Many layers in network (applies a convolution operation, reduce image (300x300 -> 10x10 example -> 1x1)
  - Goes through convolution - image, resize (completely random thing for us, but features are maintained), horses are smaller
  - smaller (horse, not picked them, but is fined because we already found it)
  - Move back, preserve and go this layer (how to get back, simplistic, horse to the right image)
  - SSD preserves information how to scale back (how to deconvolve) place the horse in the right place
  - One neural netwokrs (utilize features across the different layers), full-size, made smaller and smaller, additional power
  - adjust its weight, SSD
  - 1x Convolutional Neural Network
  - SSD -> Single Short MultiBox Detector
  - Ground Truth -> seperates empirical from inferred evidence
  - We add a grid over images to help predict object positions
  - A convolution operation is used with the SSD to reduce the image size
  - The overall idea for predicting with boxes and scale is -> power in number
