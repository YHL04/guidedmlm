# guidedmlm

Proof of concept uses generator to generate mask probability for each word and is trained by discriminator which predicts loss from mask probability and input

Blue: Guided MLM v1 (Uses 3x more compute for each training step)
Orange: Standard MLM

First plot: Bert Loss Curve (log scale)
Second plot: Discriminator Loss Curve (log scale)

![alt text](images/Figure_1.png)

![alt text](images/sketch.jpg)
