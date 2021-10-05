# split folders into train and test
import os
import numpy as np
import shutil
import random
import tqdm as tq
import sys


# # Creating Train / Test folders
root_dir = sys.argv[1]
annotations_dir = sys.argv[1]

test_ratio = 0.20

annotations_dir = ['helicoverpa_armigera', 'spodoptera_frugiperda']

#for folder in os.listdir(annotations_dir):
for folder in annotations_dir:
    
    # Create folders
    try:
        os.makedirs(root_dir + 'train')
    except FileExistsError:
        print('Directory exist')
        pass
        
    try:    
        os.makedirs(root_dir + 'test')
    except FileExistsError:
        print('Directory exist')
        pass
    
    folder_train = root_dir + 'train'
    folder_test = root_dir + 'test'

    files = os.listdir(root_dir + folder)
    np.random.shuffle(files) 
    train_files,  test_files = np.split(np.array(files), [int(len(files)* (1 - test_ratio))])


    train_FileNames = [root_dir + folder + '/' + name for name in train_files.tolist()]
    test_FileNames = [root_dir + folder + '/' + name for name in test_files.tolist()]

    print(folder)
    print('Total images: ', len(files))
    print('Training: ', len(train_files))
    print('Testing: ', len(test_files))

    # Copy-pasting images
    for name in tq.tqdm(train_FileNames):
        shutil.copy(name, folder_train)

    print('Train copied')
    for name in tq.tqdm(test_FileNames):
        shutil.copy(name, folder_test)
    print('Test copied')
