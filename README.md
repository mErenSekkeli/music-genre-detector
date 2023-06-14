# Music genre detection using KNN and STFT algorithms

This python application extracts the features of music using FFT and STFT algorithms from music genres, and then inserts it into the KNN algorithm and tries to detect music genres with Train and Test operations.

## Requirements

- Python 3.10
- Numpy
- pandas
- Sklearn
- Librosa

## Dataset

The dataset used in this project is the GTZAN dataset. This dataset contains 1000 audio tracks each 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22050Hz Mono 16-bit audio files in .wav format.

You can see the whole dataset from this link: [Whole Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

We are going to use only 5 music genre in this project and each genre has 30 music sample.
These are: metal, blues, disco, pop, jazz.

## Feature Extraction

We are going to use two different feature extraction methods in this project. These are FFT and STFT and we will use 3 different type window. These are: Hamming, Hanning and Blackman. 

Using these algorithms we extracting 9 different features from each music sample. You can see the feature types in the code or in the csv files.

## KNN Algorithm
First 20 music sample is train data and last 10 music sample is test data. We are going to use KNN algorithm to detect music genre.

## Usage

-According to usage you give k value as 3 and window type as hamming. You can change these values. Only Train parameter must '1' if you have already extracted features and you want to use only KNN algorithm.


```bash

python main.py --only_train 0 --k_value 3 --window_type hamming

```




