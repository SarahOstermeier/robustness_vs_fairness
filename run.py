import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
from UKTFACE import UKTFace
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier
from art.data_generators import PyTorchDataGenerator
from art.utils import load_mnist

from art.defences import AdversarialTrainer
# switch device to gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(model, trainloader, criterion, optimizer, epochs=5):
    train_loss = []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            inputs, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            img = model(inputs)

            loss = criterion(img, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
        print("Epoch : {}/{}..".format(e + 1, epochs),
              "Training Loss: {:.6f}".format(running_loss / len(trainloader)))
        train_loss.append(running_loss)
    plt.plot(train_loss, label="Training Loss")
    plt.show()

def main():
    # Get dataset with gender labels
    uktface = UKTFace('utkface.csv', 'UTKFace', labels='gender')



    # Won't actually train because train set is tiny!!!!
    # To split training, validation data
    l = uktface.__len__()
    training_size = int(np.floor(l * 0.3))
    validation_size = int(np.ceil(l * 0.7))
    train_data, val_data = random_split(uktface, [training_size, validation_size])

    # to use a data loader
    train_loader = DataLoader(train_data, batch_size=3, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=True)

    # Need to finetune resnet18 to work with my data

    model = models.resnet18(pretrained=True)

    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False

    fc = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(512, 100)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(100, 2)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.fc = fc
    model.to(device)

    epochs = 3
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    # set up our baseline classifier
    min_pixel_value = 0
    max_pixel_value = 1
    classifier = PyTorchClassifier(
        model=model.float(),
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 200, 200),
        nb_classes=2,
    )

    ## convert dataloader into datagenerator to make the batch training possible
    from art.data_generators import PyTorchDataGenerator
    train_generator = PyTorchDataGenerator(train_loader, len(train_data), 3)

    ## Baseline model training
    classifier.fit_generator(train_generator, nb_epochs=3)
    filename_pth = 'ckpt_resnet18_gender.pth'
    classifier.save(filename_pth)

    ## Defence model training
    attack = FastGradientMethod(classifier=classifier, eps=0.2)
    adclassifier = AdversarialTrainer(classifier, attack, ratio = 0.5)
    adclassifier.fit_generator(train_generator)
    filename_pth = 'ckpt_resnet18_gender_defence.pth'
    classifier.save(filename_pth)

    # Step 5: Evaluate the ART classifier on benign test examples

    corrects = {i: 0 for i in range(2)}
    totals = {i: 0 for i in range(2)}

    corrects_adv = {i: 0 for i in range(2)}
    for data, label in val_loader:
        x_test_adv = attack.generate(data)
        predictionAd = np.argmax(classifier.predict(x_test_adv),axis=1)
        outs = classifier.predict(data)
        predictions = np.argmax(outs, axis=1)
        expected = np.asarray(np.argmax(label, axis=1))
        result = predictions == expected
        inter = predictions[result]

        resultAd = predictionAd == expected
        interAd = predictionAd[resultAd]

        unique, counts = np.unique(interAd, return_counts=True)
        d = dict(zip(unique, counts))
        for key in d.keys():
            corrects_adv[key] += d[key]

        unique, counts = np.unique(inter, return_counts=True)
        d = dict(zip(unique, counts))
        for key in d.keys():
            corrects[key] += d[key]
        unique, counts = np.unique(expected, return_counts=True)
        d = dict(zip(unique, counts))
        for key in d.keys():
            totals[key] += d[key]


    for i in range(2):
        print("Accuracy for normal images for group : %2d: %.2f "%(i,corrects[i] / totals[i]))
        print("Accuracy for adversarial images for group : %2d: %.2f " % (i, corrects_adv[i] / totals[i]))
    print("Accuracy for normal images in total :  %.2f " % (sum(corrects.values()) / sum(totals.values())))


if __name__ == "__main__":
    main()