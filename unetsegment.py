from datetime import datetime
from glob import glob

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet_dataset import CityscapesDataset, MyDataset

from unet_model import UNet
from unet_utils import upload_image

matplotlib.use("Qt5agg")

import matplotlib.pyplot as plt
import wandb

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch.__version__)
    print(device)
    print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

    runid = datetime.now().strftime(r"%y%m%d_%H%M%S")
    batch_size = 1
    batch_size_val = 1
    epochs = 50
    lr = 0.001
    weight_decay = 0.0
    h = 1024
    w = 2048

    # wandb.init(
    #     project="vi2023_segmentering",
    #     # track hyperparameters and run metadata
    #     config={
    #         "architecture": "UNet",
    #         "dataset": "Cityscapes",
    #         "epochs": epochs,
    #         "learning_rate": lr,
    #         "weight_decay": weight_decay,
    #         "batch_size": 3,
    #         "batch_size_val": 1
    #     }
    # )
    # with open(__file__) as f:
    #     wandb.save(f.name, policy="now")

    train_path = glob("data/leftImg8bit/train/*/*leftImg8bit.png")#[:100]
    vaild_path = glob("data/leftImg8bit/val/*/*leftImg8bit.png")#[:100]
    gt_train_path = glob("data/gtFine/train/*/*gtFine_color.png")
    gt_valid_path = glob("data/gtFine/val/*/*gtFine_color.png")
    gt_gray_train_path = glob("data/gtFine/train/*/*labelIds.png")
    gt_gray_valid_path = glob("data/gtFine/val/*/*labelIds.png")
    """
    train_path = glob('/cluster/projects/vc/data/ad/open/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/*/*leftImg8bit.png')#[:100]
    vaild_path = glob('/cluster/projects/vc/data/ad/open/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/val/*/*leftImg8bit.png')#[:100]
    gt_train_path = glob('/cluster/projects/vc/data/ad/open/Cityscapes/gtFine_trainvaltest/gtFine/train/*/*gtFine_color.png')
    gt_valid_path = glob('/cluster/projects/vc/data/ad/open/Cityscapes/gtFine_trainvaltest/gtFine/val/*/*gtFine_color.png')
    gt_train_path = glob('/cluster/projects/vc/data/ad/open/Cityscapes/gtFine_trainvaltest/gtFine/train/*/*gtFine_labelIds.png')
    gt_valid_path = glob('/cluster/projects/vc/data/ad/open/Cityscapes/gtFine_trainvaltest/gtFine/val/*/*gtFine_labelIds.png')
    """
    assert len(gt_train_path) > 0

    ### Define data transformations if needed 2048 X 1024
    round_to = lambda x, mod: int(round(x/mod)*mod)

    ### Create instances of your dataset for training and validation
    train_data = CityscapesDataset("train", train_path, gt_gray_train_path, group_labels=True)
    val_data = CityscapesDataset("val", vaild_path, gt_gray_valid_path, group_labels=True)

    ### Creating the DataLoaders
    train_loader = DataLoader(train_data, batch_size, pin_memory=True, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size_val, pin_memory=True, shuffle=True, num_workers=0)

    ### initializing the model
    model = UNet(input_dim=3, output_dim=train_data.num_classes).float().to(device)
    print(f"Model weights: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    #checkpoint = torch.load("weights/unetsegment_final.pt")
    #model.load_state_dict(checkpoint["model_state_dict"])
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    steps = [10, 40, 100, 200, 300, 400, 600, 800, 1000, 1500, 2000, 2500]
    for epoch in range(1, epochs+1):
        trainloss = 0
        valloss = 0
        j = -1

        step = -1
        for img_in, label_in in tqdm(train_loader, smoothing=0.1):
            """
                Traning the Model.
            """
            #if j > 10:
            #    break
            j += 1
            step += 1
            if step in steps:
                steps.remove(step)
                break

            optimizer.zero_grad()
            img = img_in.to(device)
            label = label_in.to(device)
            output = model(img)

            loss = lossfunc(output, label)

            loss.backward()
            optimizer.step()
            trainloss += loss.item()


        #train_loss.append(trainloss/len(train_loader))
        train_loss.append(trainloss/j)
        j = -1

        for img_in, label_in in tqdm(val_loader, smoothing=0.1):
            """
                Validation of Model.
            """
            j += 1
            if j > 10:
                break

            with torch.no_grad():
                img = img_in.to(device)
                label = label_in.to(device)
                output = model(img)
                loss = lossfunc(output, label)
                valloss += loss.item()

        #val_loss.append(valloss/len(val_loader))
        val_loss.append(valloss/j)
        #writer.add_scalar("Validation Loss", val_loss / len(valid_loader), i)
        print("epoch : {} ,train loss : {} ,valid loss : {} ".format(epoch, train_loss[-1], val_loss[-1]))
        
        output = val_data.classes_to_rgb(output)
        label = val_data.classes_to_rgb(label)

        imgs = upload_image(img, output, label)
        names = ["img", "out", "lab"]
        for k in range(len(imgs)):
            imgs[k].image.save(f"output/{runid}_e{epoch}_{names[k]}.png")
            ax[k].imshow(np.array(imgs[k].image))
            #imgs[i].image.show()
        fig.canvas.draw()
        fig.canvas.flush_events()
        fig.show()
        # wandb.log({
        #     "val_loss": val_loss[-1],
        #     "train_loss": train_loss[-1],
        #     "images": upload_image(img, output, label)
        # })

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss[-1],
        }, f"weights/unetsegment_{runid}_checkpoint_{epoch+1}.pt")

    wandb.finish()