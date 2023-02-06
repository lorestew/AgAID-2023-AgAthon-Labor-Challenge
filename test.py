import os
import numpy as np
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as F

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from torchvision.utils import draw_segmentation_masks
import transforms as T
import utils

from engine import train_one_epoch, evaluate
from load import get_model_instance_segmentation, Minneapple, get_transform

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def eval():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2

    dataset = Minneapple('appletrain', get_transform(train=True))
    dataset_test = Minneapple('appletrain', get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    
    img, _ = dataset_test[0]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

        #display image
        plt.imshow(Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy()))
        plt.show()
        
        #display masks
        for i in range(len(prediction[0])):
            plt.imshow(Image.fromarray(prediction[0]['masks'][i, 0].mul(255).byte().cpu().numpy()))
            plt.show()

eval()
