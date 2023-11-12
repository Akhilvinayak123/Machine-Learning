from PIL import Image
import os


def resize(im, new_width):
    width, height = im.size
    ratio = height / width
    new_height = int(ratio * new_width)
    resized_image = im.resize((new_width, new_height))
    return resized_image




files = os.listdir("Images")
extensions = ['jpg', 'jpeg', 'png', 'glf']
for file in files:
    ext = file.split(".")[-1]
    if ext in extensions:
        im = Image.open("Images/"+file)
        im_resized = resize(im, 600)
        filepath = f"images/{file}.png"
        im_resized.save(filepath)


#im = Image.open("Images/121212.png")
#im_resized = resize(im, 600)
#im_resized.show()



