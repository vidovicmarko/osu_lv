import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

img = plt . imread ("road.jpg")

plt.figure()
plt.imshow(img, alpha=0.5)
plt.title("Posvijetljena slika")
plt.show()

print(img.shape)
height, widht, random = img.shape
column = widht // 4
plt.imshow(img[:height, column:column*2])
plt.show()

rotate = np.rot90(img, 3)
plt.imshow(rotate)
plt.title("Rotirana slika")
plt.show()

flip = np.flip(img, 1)
plt.imshow(flip)
plt.title("Zrcaljena slika")
plt.show()