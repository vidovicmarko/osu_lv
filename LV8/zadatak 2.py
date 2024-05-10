import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt

model = keras.models.load_model("model zadatak 1.keras")
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# slike trebaju biti (28, 28, 1)
x_train_s = np.expand_dims(x_train_s, -1)
x_test_s = np.expand_dims(x_test_s, -1)


y_pred = model.predict(x_test_s)
pred_labels = np.argmax(y_pred, axis=1)
wrong_labels = np.where(pred_labels != y_test)[0]

for i in range(3):
    plt.figure()
    index = wrong_labels[i]
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')  
    plt.title(f'True Label: {y_test[index]}, Predicted Label: {pred_labels[index]}')
plt.show()