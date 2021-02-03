import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

def load_images_from_folder(root_dir):
    i=0
    fn = []
    fig = plt.figure(figsize=(9,9))
    for filename in glob.iglob(root_dir + '**/*.jpg', recursive=True):
        if i < 9:
            fn = fn + [filename]
            i = i + 1
    
    img_h1 = cv2.hconcat([mpimg.imread(fn[0]),mpimg.imread(fn[1]),mpimg.imread(fn[2])])
    img_h2 = cv2.hconcat([mpimg.imread(fn[3]),mpimg.imread(fn[4]),mpimg.imread(fn[5])])
    img_h3 = cv2.hconcat([mpimg.imread(fn[6]),mpimg.imread(fn[7]),mpimg.imread(fn[8])])
    
    img_tile = cv2.vconcat([img_h1,img_h2,img_h3])                      
                          
    plt.imshow(img_tile)
    plt.show()
           
            
            
            