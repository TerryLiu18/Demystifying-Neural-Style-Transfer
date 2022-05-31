import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

pic_dir = './alpha/'

pic000 = plt.imread(pic_dir + '000.jpg')
pic025 = plt.imread(pic_dir + '025.jpg')
pic050 = plt.imread(pic_dir + '050.jpg')
pic075 = plt.imread(pic_dir + '075.jpg')
pic100 = plt.imread(pic_dir + '100.jpg')


# concat two pictures horizontally
def concat_horizontal(pic1, pic2):
    pic1 = np.array(pic1)
    pic2 = np.array(pic2)
    pic = np.concatenate((pic1, pic2), axis=1)
    return pic

# display a picture with matplotlib
def display_pic(pic, save_name):
    
    plt.imshow(pic)
    plt.imsave('img/' + save_name + '.png', pic, dpi=1200)
    plt.show()
    plt.close()

pic_1 = concat_horizontal(pic000, pic025)
pic_2 = concat_horizontal(pic000, pic025)
pic_3 = concat_horizontal(pic000, pic050)
pic_4 = concat_horizontal(pic000, pic075)
pic_5 = concat_horizontal(pic000, pic100)

display_pic(pic_1, 'pic_1')
display_pic(pic_2, 'pic_2')
display_pic(pic_3, 'pic_3')
display_pic(pic_4, 'pic_4')
display_pic(pic_5, 'pic_5')

# generate gif images using imageio
pic_list = []
for i in range(1, 6):
    pic_list.append('img/pic_' + str(i) +'.png')

# imageio.mimsave('alpha.gif', pic_list, fps=2)
imageio.mimsave('img/alpha.gif', [imageio.imread(pic) for pic in pic_list], fps=1)