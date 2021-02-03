import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def load_images_from_folder(root_dir):
    i=0
    for filename in glob.iglob(root_dir + '**/*.jpg', recursive=True):
        while i < 10:
            img = mpimg.imread(filename)
            plt.imshow(img)
            plt.show()
            i += 1