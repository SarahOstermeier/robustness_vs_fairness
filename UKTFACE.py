from __future__ import print_function, division
import csv
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image
import os
import torchvision.models as models
from collections import OrderedDict
from torch import nn, optim

# switch device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg

[age] is an integer from 0 to 116, indicating the age
[gender] is either 0 (male) or 1 (female)
[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
[date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace

Throwing out images due to missing data: 
20170109142408075.jpg.chip.jpg
20170109150557335.jpg.chip.jpg
20170116174525125.jpg.chip.jpg


'''

"""UTKFace Dataset"""


def make_csv(directory, outputfile):
    directory = 'UTKFace'

    data = []

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            data.append(filename)
            continue
        else:
            continue

    with open(outputfile, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'age', 'gender', 'race', 'datetime'])
        for entry in data:
            temp = entry.split('_')
            if (len(temp) == 4):

                age = int(temp[0])
                gender = int(temp[1])
                race = int(temp[2])
                datetime = temp[3].split('.')[0]
            writer.writerow([entry, age, gender, race, datetime])


class UKTFace(Dataset):
    def __init__(self, csv_file, root_dir, labels='gender'):
        make_csv("UTKFace", "utkface.csv")
        self.root_dir = root_dir
        self.data_info = pd.read_csv(csv_file)
        self.transformations = \
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])

        self.data_len = len(self.data_info.index)

        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.age_arr = np.asarray(self.data_info.iloc[:, 1])
        self.gender_arr = np.asarray(self.data_info.iloc[:, 2])
        self.race_arr = np.asarray(self.data_info.iloc[:, 3])
        self.datetime_arr = np.asarray(self.data_info.iloc[:, 4])
        self.labels = labels

    def __getitem__(self, index):
        # Get image name from the pandas df
        img_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(os.path.join(self.root_dir, img_name))

        # get image as tensor
        img_as_tensor = self.transformations(img_as_img)

        # labels
        age = self.age_arr[index]
        gender = self.gender_arr[index]
        race = self.race_arr[index]
        datetime = self.datetime_arr[index]
        if (self.labels == 'gender'):
            label = gender
        elif (self.labels == 'age'):
            label = age
        elif(self.labels == 'race'):
            label = race

        label = nn.functional.one_hot(torch.tensor(label), 2)

        return img_as_tensor, label

    def get_age(self, index):
        age = self.age_arr[index]
        return age

    def get_gender(self, index):
        gender = self.gender_arr[index]
        return gender

    def get_race(self, index):
        race = self.race_arr[index]
        return race

    def show_image(self, index):
        img_name = self.image_arr[index]
        img_as_img = Image.open(os.path.join(self.root_dir, img_name))
        plt.imshow(img_as_img)
        plt.axis('off')
        plt.show()

    def __len__(self):
        return self.data_len


def main():
    # Get dataset
    uktface = UKTFace('utkface.csv', 'UTKFace', labels='gender')


if __name__ == "__main__":
    main()
