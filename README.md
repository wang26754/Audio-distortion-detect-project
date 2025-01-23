The Sound Event Detection (SED) approach described consists of two main stages: the sound representation
stage and the framewise classification stage. In the sound representation stage, frame-level
acoustic features are extracted from the acoustic signal to form a feature matrix X ∈ RF×T , where F
is the number of features per frame and T is the number of frames in the signal.
The classification stage involves estimating the probabilities p(y | X, θ) for the target output vector
y ∈ RT , where y represents the probability of the target event in each frame and θ are the classifier
parameters. The classifier parameters are trained using supervised learning, with target outputs yt set
to 1 if the event is present in frame t and 0 otherwise. These probabilities are binarized, typically with
a threshold of 0.5, to obtain binary event activity predictions ˆy.
