from PIL import Image
import subprocess
from keras.models import Sequential
from keras.layers import Dense, Convolution2D,Dense,Dropout,Flatten
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from keras.models import load_model
import tkinter as tk
from tkinter.messagebox import showinfo


image = Image.new('RGB', (100,100),"black")
image.save("image.png", "PNG")
class_names = {0: "book", 1: "clock", 2: "hand", 3: "car", 4: "fish", 5: "laptop", 6: "cup"}


p = subprocess.call(
    ["/usr/bin/open", "-W", "-n", "image.png"]
    )

img = Image.open("image.png", 'r')
img= img.convert('L') # convert image to monochrome
img = img.resize((28,28),Image.ANTIALIAS)
img.save("image_new.png", "PNG")
# img.save("image_new.png", dpi=(600,600))
img = Image.open("image_new.png", 'r')


img = np.array(img)
img = img.astype('float32') / 255.  ##scale images
plt.imshow(img)
plt.show()

img = np.expand_dims(img, axis=2)
img = np.expand_dims(img, axis=0)

model = load_model("digit_model.h5")
# reading output

op = model.predict(img)


print("The object drawn is :",class_names[np.argmax(op)])
# messagebox.showinfo("OUTPUT", "The drawn object is: "+str(class_names[np.argmax(op)]))
root = tk.Tk()
root.withdraw()
showinfo(title="OUTPUT", message="The drawn object is: "+str(class_names[np.argmax(op)]))
