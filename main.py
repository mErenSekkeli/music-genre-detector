import os
import numpy as np
import wave
import csv
import argparse
from tqdm import tqdm
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def fft(x):
    N = len(x)
    
    if N <= 1:
        return x
    # Divide and conquer
    even = fft(x[0::2])
    odd = fft(x[1::2]) 
    
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    
    return np.concatenate([even + factor[:N // 2] * odd,
                           even + factor[N // 2:] * odd])


def stft(x, window_size, hop_size, window_type):
    N = len(x)

    if window_type == "hamming":
        window = np.hamming(window_size)
    elif window_type == "hanning":
        window = np.hanning(window_size)
    elif window_type == "blackman":
        window = np.blackman(window_size)
    else:
        raise ValueError("Window type must be one of hamming, hanning, blackman")

    num_frames = (N - window_size) // hop_size + 1
    
    stft_matrix = np.zeros((window_size, num_frames), dtype=np.complex64)
    target_length = get_length_padding(window_size)
    
    for i in tqdm(range(num_frames)):
        start = i * hop_size
        end = start + window_size
        frame = x[start:end]

        windowed_frame = frame * window

        add_padding(windowed_frame, target_length)

        stft_matrix[:, i] = fft(windowed_frame)
    
    return stft_matrix

def add_padding(signal, target_length):
    current_length = len(signal)
    if current_length >= target_length:
        return signal[:target_length]
    else:
        padding_length = target_length - current_length
        padding = np.zeros(padding_length)
        return np.concatenate((signal, padding))


def get_length_padding(size):
    tmp = 1
    while tmp < size:
        tmp *= 2
    return tmp

#First Parameter
def get_frequency_power(fft_result):
    return np.sum(np.abs(fft_result) ** 2)

#Second Parameter
def get_amplitude_mean(fft_result):
    return np.mean(np.abs(fft_result))

#Third Parameter
def get_weighted_frequency_average(magnitudes):
    weighted_sum = np.sum(np.arange(len(magnitudes)) * magnitudes)
    
    sum_magnitudes = np.sum(magnitudes)
    
    weighted_average = weighted_sum / sum_magnitudes
    
    return weighted_average

def extract_features(stft_result):
    frequency_power, amplitude_mean, weighted_frequency = extract_fft_features(stft_result)

    frequency_power_features = extract_triple_features(frequency_power)
    amplitude_mean_features = extract_triple_features(amplitude_mean)
    weighted_frequency_features = extract_triple_features(weighted_frequency)

    return [frequency_power_features, amplitude_mean_features, weighted_frequency_features]

def extract_fft_features(stft_result):
    frequency_power = []
    amplitude_mean = []
    weighted_frequency = []
    for fft_result in stft_result:
        frequency_power.append(get_frequency_power(fft_result))
        amplitude_mean.append(get_amplitude_mean(fft_result))
        weighted_frequency.append(get_weighted_frequency_average(frequency_power))

    return frequency_power, amplitude_mean, weighted_frequency
        
def extract_triple_features(feature):
    mean_feature = np.mean(feature)
    median_feature = np.median(feature)
    deviation_feature = np.std(feature, ddof=1)
    return [mean_feature, median_feature, deviation_feature]

def create_identity_to_voices(filename):
    number_string = filename.split(".", 1)[-1].split(".")[0]
    try:
        number = int(number_string)
        label = filename.split(".")[0]
        combined_string = label + "_" + str(number)
        return combined_string, label
    except ValueError:
        return None

def create_csv(fileName,identifier, features, mode):
    with open(fileName, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        if mode == "features":
            for i in range(len(identifier)):
                writer.writerow([identifier[i][0], features[i][0][0], features[i][0][1], features[i][0][2], features[i][1][0],features[i][1][1],features[i][1][2], features[i][2][0], features[i][2][1], features[i][2][2], identifier[i][1]])
        elif mode == "datasets":
            for i in range(len(identifier)):
                writer.writerow([identifier[i][0], features[i][0], features[i][1], features[i][2], features[i][3], features[i][4], features[i][5], features[i][6], features[i][7],  features[i][8], identifier[i][1]])
        elif mode == "test":
            for i in range(len(identifier)):
                writer.writerow([identifier[i][0], features[i], identifier[i][1]])
        csvfile.close()    

def constructorFunc(window_type):
    stft_result = stft(audio_data, 1024, 512, window_type)
    features = extract_features(stft_result)
    identifier = create_identity_to_voices(filename)
    return identifier, features

def initializeCsv(fileName, mode):
    with open(fileName, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        if mode == "features" or mode == "datasets":
            writer.writerow(['id', 'frequency_power_mean', 'frequency_power_median', 'frequency_power_deviation', 'amplitude_mean_mean', 'amplitude_mean_median', 'amplitude_mean_deviation', 'weighted_frequency_mean', 'weighted_frequency_median', 'weighted_frequency_deviation', 'label'])
        elif mode == "test":
            writer.writerow(['id','predicted_label', 'expected_label'])
        csvfile.close()

def trainModel(k_value, window_type, features_name):
    data = pd.read_csv(features_name, delimiter=';')

    features = data.iloc[:, 1:-1].values
    labels =  data.iloc[:, -1].values
    id = data.iloc[:, 0].values
    
    num_features_per_category = 30
    X_train = np.concatenate([features[i:i+20] for i in range(0, len(features), num_features_per_category)])
    y_train = np.concatenate([labels[i:i+20] for i in range(0, len(labels), num_features_per_category)])
    id_train = np.concatenate([id[i:i+20] for i in range(0, len(id), num_features_per_category)])

    train_name = window_type.upper() + "_TRAIN.csv"
    initializeCsv(train_name, "datasets")
    create_csv(train_name, list(zip(id_train,y_train)), X_train, "datasets")

    X_test = np.concatenate([features[i+20:i+30] for i in range(0, len(features), num_features_per_category)])
    y_test = np.concatenate([labels[i+20:i+30] for i in range(0, len(labels), num_features_per_category)])
    id_test = np.concatenate([id[i+20:i+30] for i in range(0, len(id), num_features_per_category)])

    test_name = window_type.upper() + "_TEST.csv"
    initializeCsv(test_name, "datasets")
    create_csv(test_name, list(zip(id_test,y_test)), X_train, "datasets")

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    knn = KNeighborsClassifier(n_neighbors=int(k_value))
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    result_name = "K_" + str(k_value) + "_" +  window_type.upper() + "_RESULT.csv"
    initializeCsv(result_name, "test")
    create_csv(result_name, list(zip(id_test,y_test)), y_pred, "test")
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy: ", accuracy)


def deleteFeaturesCsv(filename):
    os.remove(filename)

def read_wav_file(filename):
    with wave.open(filename, 'rb') as wav:
        num_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        num_frames = wav.getnframes()
        audio_data = wav.readframes(num_frames)
        
        audio_data = np.frombuffer(audio_data, dtype=np.int16)
        
    return audio_data, num_channels, sample_width, sample_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--only_train', default=0)
    parser.add_argument('--k_value', default=3)
    parser.add_argument('--window_type', default='hamming')
    args = parser.parse_args()
    features_name = args.window_type.upper() + "_FEATURES.csv"
    if int(args.only_train) == 0:
        data_path = 'Data/'
        music_genres = ['metal', 'blues', 'disco', 'jazz', 'pop']
        initializeCsv(features_name, "features")
        for genre in tqdm(music_genres):
            print('\n' + genre)
            identifier = []
            all_features = []
            new_data_path = data_path + genre + '/'
            for filename in tqdm(os.listdir(new_data_path)):
                file_path = os.path.join(new_data_path, filename)
                audio_data, num_channels, sample_width, sample_rate = read_wav_file(file_path)
                tmp_identifier, tmp_all_features = constructorFunc(args.window_type)
                identifier.append(tmp_identifier)
                all_features.append(tmp_all_features)

            create_csv(features_name, identifier, all_features, "features")

    trainModel(args.k_value, args.window_type, features_name)