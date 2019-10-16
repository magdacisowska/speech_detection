from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def read_input_output(filename, end):
    input_data = open("inputs/" + filename + ".txt")
    lines = input_data.readlines()
    x = []
    for line in lines:
        x.append(line.split())
    x = np.array(x[:end])
    x = x.astype(np.float)

    output_data = open("labels/giant.txt")
    lines = output_data.readlines()
    y = []
    for line in lines:
        y.append(line[:-1].split())
    y = y[20:end+20]
    y = np.array(y).reshape(end, 3)
    y = y.astype(np.float)

    return x, y


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(533, )),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

x, y = read_input_output('giant_input', 99980)

learning_rate = 0.001
momentum = 0.9
batch_size = 50
num_epochs = 10

optimizer = keras.optimizers.SGD(lr=learning_rate, decay=0.1, momentum=momentum, nesterov=False)

model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(x, y, epochs=num_epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

# performance plots
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
