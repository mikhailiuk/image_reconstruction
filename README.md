# Image interpolation with auto-encoder.

Real world data is noisy. Noise can come in the form of incorrect or missing values. This code is supplementary to an article [1], describing a way of filling missing values with auto-encoders. An auto-encoder is trained on noisy data with no reference values available for missing entries. The procedure is explained on the task of image reconstruction. If using the code please cite [2]

[1] https://medium.com/@aliaksei.mikhailiuk/unsupervised-learning-for-data-interpolation-e259cf5dc957

[2] Mikhailiuk A. and Faul A., 2018. "Deep Learning Applied to Seismic Data Interpolation". In: European Association of Geoscientists and Engineers (EAGE) 2018. IEEE 

In order to run the code:

1) specify train, test and validation images in ./data/dataset.csv
	- can have multiple images (every line in the file corresponds to a new image)
	- The format: 
		- image_noisy: image to train on (e.g. ./data/noisy_images/0_70.bmp is simply 0_70), 0 - image number, 70 - amount of noise
		- mask_train: mask for the noisy image - specifies position of the missing values
		- mask_validation: validation mask
		- image_test: test image to reconstruct with the trained network
		- mask_test: mask specifiying missing positions of pixels in the test image

2) Activate tensorflow
3) python3 main.py