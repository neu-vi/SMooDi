import os
from pprint import pformat

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from statistics import mean
from torch.optim import AdamW
import torch.nn as nn
from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
from mld.models.architectures.mld_style_encoder import StyleClassification
from mld.data.humanml.data.dataset import StyleMotionDataset

def reset_loss_dict(loss_dict):
    loss_dict = {
        "crossentropy_loss": []
    }


def main():
    # create dataset
    datasets = StyleMotionDataset('train')
    test_dataset = StyleMotionDataset('test')

    dataloader = torch.utils.data.DataLoader(dataset=datasets, batch_size=128, shuffle=True,drop_last=True)
    testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    
    model = StyleClassification(nclasses=47).cuda()

    optimizer = AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    loss_dict = {
        "crossentropy_loss": [],
        "tri_loss":[]
    }


    log_freq = 2
    n_epoch = 200

    test_freq = 10
    save_freq = 100
    model_dir = "./experiments"
    model.train()
    
    for epoch in range(n_epoch):
        total = 0
        correct = 0
        for i, train_data in enumerate(dataloader):
            optimizer.zero_grad()

            motion_input = train_data['motion']

            output_score,feat = model(motion_input,stage="Both")

            _, predicted = torch.max(output_score, 1)
            total += train_data["motion"].size(0)
            correct += (predicted == train_data["label"]).sum().item()

            
            loss = criterion(output_score,train_data['label']) +0.001*center_loss(feat,train_data['label'])
            loss_all = loss
            loss_dict["crossentropy_loss"].append(loss.item())

            loss_all.backward()
            optimizer.step()

            if (i + 1) % log_freq == 0:
                print('Train: Epoch [{}/{}], Step [{}/{}]| loss: {:.4f}  accuracy: {:.4f}'.format(epoch + 1, n_epoch, i + 1, len(dataloader), mean(loss_dict["crossentropy_loss"]),100*(correct / total)))

        if (epoch + 1) % test_freq == 0:
            model.eval()
            total = 0
            correct = 0
            for i, test_data in enumerate(testloader):
                output_score = model(test_data['motion'])
                _, predicted = torch.max(output_score, 1)
                total += test_data["motion"].size(0)
                correct += (predicted == test_data["label"]).sum().item()

            print('Test: Epoch [{}/{}]| accuracy: {:.4f}%'
                                 .format(epoch + 1, n_epoch, 100*(correct / total)))
            model.train()

        if (epoch+1) % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(model_dir, "style_encoder_robust47_{}.pt".format(epoch+1)))

   


if __name__ == "__main__":
    main()
