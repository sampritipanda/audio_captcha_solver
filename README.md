# Solving Audio Captchas

Solving Audio Captchas using Machine Learning

Authors: Sampriti Panda, Duy Nguyen

## Requirements

* python >= 3.6
* numpy
* scipy
* matplotlib
* python_speech_features: `pip install python_speech_features`
* sklearn
* keras (Not-needed, you can comment out the imports)
* `pip install numba==0.48.0`
* librosa, spectrum: `pip install librosa spectrum`

## Generating Training Data

* We have provided around 20 train and 10 test cases per category, but you need to generate around 1000 train data to replicate our results.
* To generate data using our scripts, please cd into the `training_data/` directory and run: `./gen_data.sh`.
* You can also download pre-generated training data from: https://drive.google.com/file/d/19ypbdOiafc3Ocr9ltHIFjJI9uQXlEuJR/view?usp=sharing
* `poc.py` contains our original algorithm, which gives around 70% accuracy on digits and 50% on letters.
* `poc2.py` contains our improved algorithm, which gives around 95% accuracy.
* To run either of these implementations, modify the `DIR_TRAIN` and `DIR_TEST` directories to the necessary locations, and run `python poc.py`.
