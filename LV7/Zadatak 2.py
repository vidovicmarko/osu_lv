import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

# 7.5.2)

# 
colors = np.unique(img_array_aprox, axis= 0)
print(f"U ovoj slici postoji: {len(colors)} različith boja.")

#
km= KMeans(n_clusters=5, init="k-means++")
km.fit(img_array_aprox)

cluster_centers = km.cluster_centers_
labels = km.predict(img_array_aprox)

img_array_aprox = cluster_centers[labels]
img_aprox = np.reshape(img_array_aprox, (w, h, d))

plt.figure()
plt.title("Druga slika")
plt.imshow(img_aprox)
plt.tight_layout()
plt.show()

def funk(path):
    img = Image.imread(path)

    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    img = img.astype(np.float64) / 255
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))
    img_array_aprox = img_array.copy()

    km= KMeans(n_clusters=4, init="k-means++")
    km.fit(img_array_aprox)
    cluster_centers = km.cluster_centers_
    labels = km.predict(img_array_aprox)
    img_array_aprox = cluster_centers[labels]
    img_aprox = np.reshape(img_array_aprox, (w, h, d))

    plt.figure()
    plt.title("Druga slika")
    plt.imshow(img_aprox)
    plt.tight_layout()
    plt.show()

funk("imgs\\test_2.jpg")
funk("imgs\\test_3.jpg")
funk("imgs\\test_4.jpg")
funk("imgs\\test_5.jpg")
funk("imgs\\test_6.jpg")

# 6)

clusters = range(1,10)
distortions = []

for k in (clusters):
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    km.fit(img_array)
    distortions.append(km.inertia_)

plt.figure()
plt.xlabel('K')
plt.ylabel('J')
plt.plot(clusters,distortions)
plt.grid(True)
plt.show()

# 7)

labels = km.labels_

for id in range(2):  
    cluster_mask = labels.reshape(w, h) == id

    binary_image = np.zeros((w, h), dtype=np.uint8)
    binary_image[cluster_mask] = 255 

    plt.figure()
    plt.title("Binarna slika za cluster = {}".format(id))
    plt.imshow(binary_image, cmap='gray')  
    plt.tight_layout()
    plt.show()