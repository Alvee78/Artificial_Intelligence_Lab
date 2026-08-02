from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Input, Flatten
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

(tx, tl), (ttx, ttl)  = mnist.load_data()

tx = tx / 255.0
ttx = ttx / 255.0


tx = np.expand_dims(tx, axis= -1)
ttx = np.expand_dims(ttx, axis= -1)

i = Input((28,28,1))
x = Conv2D(64, 3, activation="relu")(i)
x = MaxPool2D()(x)
x = Conv2D(32, 3,activation="relu")(x)
x = MaxPool2D()(x)
x = Conv2D(16, 3,activation="relu")(x)
x = MaxPool2D()(x)
x = Flatten()(x)
x = Dense(512, activation="relu")(x)
o = Dense(10, activation="softmax")(x)
model = Model(i, o)
print(model.summary())

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss="sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

es = tf.keras.callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 2,
    restore_best_weights = True
)

h = model.fit(
    tx, tl, epochs = 100, validation_split= 0.1, callbacks=[es]
)

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.show()

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.show()

tloss, ta = model.evaluate(tx, tl)
print(ta)





