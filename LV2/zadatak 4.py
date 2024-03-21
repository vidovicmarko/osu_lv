import numpy as np
import matplotlib.pyplot as plt

zero = np.zeros((50, 50))
one = np.ones((50, 50))
topRow = np.hstack((zero, one))
bottomRow = np.hstack((one, zero))
matrix = np.vstack((topRow, bottomRow))

plt.figure()
plt.imshow(matrix, cmap="gray")
plt.show()