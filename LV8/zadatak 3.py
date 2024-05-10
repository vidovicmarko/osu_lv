import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image

model = keras.models.load_model("model zadatak 1.keras")

for i in range(10):
    image_path = f"slike/test_{i}.png"
    image = Image.open(image_path).convert('L') 
    image = image.resize((28, 28))  
    image_array = np.array(image)  

    image_array = image_array.astype("float32") / 255
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)

    predicted = np.argmax(model.predict(image_array))

    print(f"Predicted label for {i}: {predicted}")

'''
Predicted label for 0: 9
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step

Predicted label for 1: 8
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step

Predicted label for 2: 2
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step

Predicted label for 3: 9
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step

Predicted label for 4: 9
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 18ms/step

Predicted label for 5: 8
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step

Predicted label for 6: 9
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step

Predicted label for 7: 3
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 19ms/step

Predicted label for 8: 4
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 20ms/step
Predicted label for 9: 9
'''