
The Audio Distortion Detection (ADD) approach consists of two main stages: the audio representation stage and the framewise classification stage.

In the audio representation stage, frame-level acoustic features are extracted from the audio signal to form a feature matrix X, where the rows represent the number of features per frame and the columns represent the number of frames in the signal. These features capture information relevant to identifying distortions in the audio.

The classification stage involves estimating the probabilities of distortion presence in each frame, represented as p(d | X, θ), where d indicates the target distortion label, X is the feature matrix, and θ are the classifier parameters. The classifier is trained using supervised learning, with the target output for each frame set to 1 if distortion is present and 0 otherwise. These probabilities are binarized using a threshold, typically 0.5, to produce binary distortion activity predictions.
