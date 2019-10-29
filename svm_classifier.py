import numpy as np
import datetime
from math import log, sqrt
from sklearn import svm, metrics


def read_input_output(label_filename, filename, end):
    input_data = open("inputs/" + filename + ".txt")
    lines = input_data.readlines()[:end]
    x = []
    for line in lines:
        x.append(line.split())
    x = np.array(x[:end])
    x = x.astype(np.float)

    output_data = open("labels/" + label_filename + ".txt")
    lines = output_data.readlines()[:end]
    y = []
    for line in lines:
        cls = 0 if line == '0 1\n' else 1
        y.append(cls)

    return x, y


def signal_energy(frame):
    return log(sum(frame ** 2))


def zero_crossing_rate(frame):
    return ((frame[:-1] * frame[1:]) < 0).sum()


def norm_autocorr_coef(frame):
    num = sum([(frame[n] * frame[n+1]) for n in range(len(frame) - 1)])
    standard_frame = frame[:len(frame) - 1]
    lag_frame = frame[2:]
    den = sqrt((sum(standard_frame ** 2)) * (sum(lag_frame ** 2)))
    return num/den


def lin_pred(frame, rank):
    R = np.eye(rank)                    # auto-correlation matrix
    a = np.zeros(shape=(rank, 1))       # parameters vector [a_1, a_2, ... , a_r]'
    R_vec = np.zeros(shape=(rank, 1))   # auto-correlation vector [R(1), R(2), ... , R(r)]'

    # fill auto-correlation matrix
    for i in range(rank):
        vector_1 = frame[:rank]
        vector_2 = frame[i + 1:rank + i + 1]
        R_vec[i] = sum(np.convolve(vector_1, vector_2))
        for j in range(rank):
            lag = np.abs(j - i)
            vector_2 = frame[lag:rank+lag]
            R[i][j] = sum(np.convolve(vector_1, vector_2))

    # calculate model(filter) coefficients(poles)
    try:
        a = np.dot(np.linalg.inv(R), R_vec).reshape((1, rank))[0]
    except np.linalg.LinAlgError:
        return np.ones(rank).reshape((1, rank)), 10         # in case of singular matrix (very rare)

    # calculate inverse filter coefficients
    a_inverse = np.ones(rank + 1)
    a_inv = [-a[j] for j in range(rank)]
    a_inverse[1:] = a_inv

    # calculate inverse filter response (error)
    inverse_filter_response = np.convolve(frame, a_inverse)
    error = abs(sum(inverse_filter_response))

    return a[1], error


def features_from_file(filename):
    index = 0
    frames = open('inputs/' + filename + '.txt').readlines()
    for frame in frames:
        frame = frame.split()
        frame = [float(frame[i]) for i in range(len(frame))]
        feature_1 = signal_energy(np.array(frame))
        feature_2 = zero_crossing_rate(np.array(frame))
        feature_3 = norm_autocorr_coef(np.array(frame))
        feature_4, feature_5 = lin_pred(frame, 12)

        feature_vec = np.array([feature_1, feature_2, feature_3, feature_4, feature_5]).reshape((1, 5))
        with open('inputs/svm_input_' + filename + '.txt', 'a+') as f:
            try:
                np.savetxt(f, feature_vec)
            except:
                np.savetxt(f, np.array([0, 0, 0, 0, 0]).reshape((1, 5)))    # very rare
        print(index)
        index += 1


def to_srt(filename):
    '''Create .srt file with subtitles'''
    file = open(filename)
    lines = file.readlines()
    i = 0
    for line in lines:
        with open('svm_output_2.srt', 'a+') as f:
            f.write(str(i) + '\n')
            time_stamp = datetime.timedelta(milliseconds=i * 10).__str__()[:11] + ' --> ' + \
                         datetime.timedelta(milliseconds=i * 10 + 10) .__str__()[:11] + '\n'
            time_stamp = time_stamp.replace('.', ',')
            f.write(time_stamp)
            if line == 'NO SPEECH\n':
                f.write('NO SPEECH\n\n')
            else:
                f.write('SPEECH\n\n')

            i += 1
            print(i)


x, y = read_input_output(filename='svm_input_ugandan', label_filename='ugandan', end=16000)
x_2, y_2 = read_input_output(filename='svm_input_split_2PpxiG0WU18', label_filename='split_2PpxiG0WU18', end=16000)
x_3, y_3 = read_input_output(filename='svm_input_split_4gVsDd8PV9U', label_filename='split_4gVsDd8PV9U', end=16000)

classifier = svm.SVC(gamma='scale', verbose=True)
classifier.fit(x_2[4000:], y_2[4000:])
y_test = classifier.predict(x_2[:4000])
print("Accuracy:", metrics.accuracy_score(y_test, y_2[:4000]))
with open('svm_results_2.txt', 'w+') as file:
    for item in y_test:
        if item == 0:
            file.write("SPEECH\n")
        else:
            file.write("NO SPEECH\n")
to_srt('svm_results_2.txt')
# features_from_file('split_2PpxiG0WU18')
# features_from_file('split_2qQs3Y9OJX0')
# features_from_file('split_4gVsDd8PV9U')

