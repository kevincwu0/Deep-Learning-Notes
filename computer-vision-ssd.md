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
