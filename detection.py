import librosa
import librosa.display
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

WINDOW_HOP = 0.01       # [sec]
WINDOW_SIZE = 0.025     # [sec]


def get_file_labels(filename):
    file_data = speech_labels_data[speech_labels_data['filename'] == filename]
    start_times = file_data['start'].to_list()
    end_times = file_data['stop'].to_list()
    labels = file_data['type'].to_list()

    i = 0
    labels_to_csv = []
    for n_window in range(0, 89998):
        start_t = n_window * WINDOW_HOP
        if i < len(start_times) - 1 and start_t >= start_times[i+1] - 900:
            i += 1

        if start_t >= start_times[i] - 900: # and start_t + WINDOW_SIZE <= end_times[i] - 900:
            labels_to_csv.append(labels[i])

    with open('labels/giant.txt', 'a+') as f:
        for label in labels_to_csv:
            if label == 'CLEAN_SPEECH':
                c_label = '1 0 0'
            if label == 'NO_SPEECH':
                c_label = '0 1 0'
            if label == 'SPEECH_WITH_NOISE':
                c_label = '0 0 1'
            f.write(c_label + "\n")
        print(name + " labels saved")


def get_file_mfcc(filename):
    y, fs = librosa.load('audio_dataset/' + filename + '.wav', sr=None)

    mel_specgram = librosa.feature.melspectrogram(y, sr=fs, n_mels=26, hop_length=int(WINDOW_HOP * fs),
                                                  win_length=int(WINDOW_SIZE * fs))
    mfcc_s = librosa.feature.mfcc(S=librosa.power_to_db(mel_specgram), n_mfcc=26, sr=fs)
    mfcc_s = np.reshape(mfcc_s[:13], newshape=(len(mfcc_s.T), 13))
    mfcc_s = np.array(mfcc_s)

    scaler = StandardScaler()
    normalized_mfcc_s = scaler.fit_transform(mfcc_s)

    with open('inputs/giant.txt', 'ab') as f:
        np.savetxt(f, normalized_mfcc_s)
        print(name + " mfcc saved")


speech_labels_data = pd.read_csv('ava_speech_labels_v1.csv')
with open('test_labels') as file:
    lines = file.readlines()
    for filename in lines:
        name = filename[:11]            # discard file extension
        get_file_labels(name)
        get_file_mfcc(name)
