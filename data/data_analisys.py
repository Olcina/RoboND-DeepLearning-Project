# Training set images analysis

#Load modules
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
from  tqdm import tqdm

# TODO: Load images filenames in list
def img_to_list(folder):
    img_list = glob.glob(folder + '*.png')
    return img_list



def random_image_show(img):
    red_channel = img.copy()
    red_channel[:,:,1] = 0
    red_channel[:,:,2] = 0
    green_channel = img.copy()
    green_channel[:,:,0] = 0
    green_channel[:,:,2] = 0
    blue_channel = img.copy()
    blue_channel[:,:,1] = 0
    blue_channel[:,:,0] = 0
    # img_layer1 = np.concatenate((img[:,:,0],border,img[:,:,1],border,img[:,:,2]),axis=1)
    border = np.ones((img.shape[0],10,3))
    img_concat = np.concatenate((red_channel,border,blue_channel,border,green_channel),axis=1)
    inp = input("Do you want to show the image (y/n) ")
    if inp == 'y':
        plt.figure(figsize=(10,5))
        plt.imshow(img_concat)
        plt.show()
    else:
        print('end run')

# TODO: load images in numpy number
# TODO: localize blue pixels in images == Hero
# TODO: localize green pixels in images == People

# TODO: Visualization



def channel_stats(img,channel):
    # channels = RGB (0,1,2)
    cha = img[:,:,channel]
    px_total = img.shape[0] * img.shape[1]
    px_count = np.count_nonzero(cha)
    percent = px_count/px_total
    return px_count, percent


def normalize(img):
    x = img/255
    return x

def unnormalize(img):
    x = img*255
    return x

def world_analisys(img):
    #analize all channel_stats and append to a boolean list
    # world = [hero,people]
    world = []
    hero_px = channel_stats(img,2)
    people_px= channel_stats(img,1)
    # the hero is in the blue channel
    if hero_px[0] > 0:
        world.append(True)
    else:
        world.append(False)
    # the people is in the green channel
    if people_px[0] > 0:
        world.append(True)
    else:
        world.append(False)

    return world

def analisys():
    images = img_to_list('train/train_combined/masks/')
    hero_count = 0
    people_count = 0
    both_count = 0
    total_img = 0
    for img_file in tqdm(images, desc= 'images read', leave=False, ncols=60):
        img = np.array(mpimg.imread(img_file))
        sel = world_analisys(img)
        if sel[0] == True:
            hero_count += 1
            if sel[1] ==True:
                both_count += 1

        if sel[1] ==True:
            people_count += 1
        total_img += 1

    print('Hero :',hero_count)
    print('People :',people_count)
    print('Both: ',both_count)
    print('Total: ',total_img)

#Run
if __name__ == '__main__':
    #load all images in list
    # analisys
    images = img_to_list('train/train_combined/masks/')
    # select 1 random images
    ran_img = random.choice(images)
    img = np.array(mpimg.imread(ran_img))
    mean = np.mean(normalize(img))
    print('img mean channel 0', mean)
    #read the image
    new_ima = normalize(img)

    #output the shape
    # print('image shape: ',img.shape)

    #analize the image
    #extrac the tree layer and print
    # print(img[:,:,0].shape)
    #Show the image if required
    random_image_show(unnormalize(new_ima))
