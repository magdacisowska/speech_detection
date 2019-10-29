from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def performance_plots(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def read_input_output(label_filename, filename, end):
    input_data = open("inputs/" + filename + ".txt")
    lines = input_data.readlines()
    x = []
    for line in lines:
        x.append(line.split())
    x = np.array(x[:end])
    x = x.astype(np.float)

    output_data = open("labels/" + label_filename + ".txt")
    lines = output_data.readlines()
    y = []
    for line in lines:
        y.append(line[:-1].split())
    y = np.array(y).reshape(end, 2)
    y = y.astype(np.float)

    return x, y


# --- normalized input
# x_train, y_train = read_input_output(label_filename='giant_input', filename='giant_input_normalized', end=169940)
# x_train, y_train = read_input_output(label_filename='medium_input', filename='medium_input_normalized', end=50000)
# x_test, y_test = read_input_output(label_filename='small_input', filename='small_input_normalized', end=16980)

# --- standarized input
# x_train, y_train = read_input_output(label_filename='giant_input', filename='giant_input_standarized', end=169940)
# x_train, y_train = read_input_output(label_filename='medium_input', filename='medium_input_standarized', end=50000)
# x_test, y_test = read_input_output(label_filename='small_input', filename='small_input_standarized', end=16980)

# --- inputs by files
# x_train, y_train = read_input_output(label_filename='french_input', filename='french_input', end=89958)
# x_train, y_train = read_input_output(label_filename='polish_input', filename='polish_input', end=89958)
x_train, y_train = read_input_output(label_filename='ugandan_input', filename='ugandan_input', end=89958)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(533, )),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

# hyper parameters
learning_rate = 0.0001
momentum = 0.9
batch_size = 50
num_epochs = 30

# training
optimizer = keras.optimizers.SGD(lr=learning_rate, decay=0.1, momentum=momentum, nesterov=False)
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=0.1)
evaluation = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=1)
prediction = model.predict(x_train[:40000])

print(evaluation)
with open('results.txt', 'w+') as f:
    for line in prediction:
        if np.argmax(line) == 0:
            result = '1 0\n'
        else:
            result = '0 1\n'
        f.write(result)

performance_plots(history)
