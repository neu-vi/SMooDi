import numpy as np
import torch

from mld.data.humanml.scripts.motion_process import (process_file,
                                                     recover_from_ric)

from .base import BASEDataModule
from .humanml.data.dataset import Text2MotionDatasetV2, TextOnlyDataset,Text2MotionDatasetCMLD,Text2MotionDatasetCMLDTest


class HumanML3DDataModule(BASEDataModule):
    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = "humanml3d"
        self.njoints = 22
        
        self.cfg = cfg
        is_train_cmld = self.cfg.TRAIN.ABLATION.CMLD

        is_cycle = self.cfg.TRAIN.ABLATION.CYCLE
        is_test = self.cfg.TRAIN.ABLATION.TEST
        is_two_dataset = self.cfg.TRAIN.ABLATION.TWODATASET
        
        if phase == "text_only":
            self.Dataset = TextOnlyDataset
        else:
            if is_train_cmld:
                if is_cycle and is_two_dataset and is_test == False:
                    self.Dataset = Text2MotionDatasetCMLD
                elif is_test:
                    self.Dataset = Text2MotionDatasetCMLDTest
            else:
                self.Dataset = Text2MotionDataset
        
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
    
    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def feats2joints_wo_norm(self,features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean

        joints = recover_from_ric(features, self.njoints)
        joints =  (joints - mean) / std
        return joints

    def transforms(self,joints):
        mean = torch.tensor(self.hparams.mean)
        std = torch.tensor(self.hparams.std)

        joints =  (joints - mean) / std

        return joints

    def joints2feats(self, features):
        features = process_file(features, self.njoints)[0]
        # mean = torch.tensor(self.hparams.mean).to(features)
        # std = torch.tensor(self.hparams.std).to(features)
        # features = (features - mean) / std
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        # random select samples for mm
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.TEST.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
