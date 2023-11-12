from PIL import Image
import os

def resize(im, new_size):
    resized_image = im.resize(new_size, Image.LANCZOS)
    return resized_image

files = os.listdir("Images")
extensions = ['jpg', 'jpeg', 'png', 'gif']
for file in files:
    ext = file.split(".")[-1]
    if ext in extensions:
        im = Image.open("Images/"+file)
        new_size = (300, 300)  # Adjust the size as needed
        im_resized = resize(im, new_size)
        filepath = f"images/{file}"
        im_resized.save(filepath)