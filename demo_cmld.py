import logging
import os
import time
from builtins import ValueError
from multiprocessing.sharedctypes import Value
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
# from torchsummary import summary
from tqdm import tqdm
from mld.config import parse_args
# from mld.datasets.get_dataset import get_datasets
from mld.data.get_data import get_datasets
from mld.data.sampling import subsample, upsample
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
from mld.models.architectures.mld_style_encoder import StyleClassification
from mld.utils.demo_utils import load_example_input
from mld.data.humanml.utils.plot_script import plot_3d_motion
from moviepy.editor import VideoFileClip

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]

def build_dict_from_txt(filename):
    result_dict = {}
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 3:
                key = parts[2]
                value = parts[1].split("_")[0]
                result_dict[key] = value
                
    return result_dict


def convert_mp4_to_gif(input_file, output_file, resize=None):
    clip = VideoFileClip(input_file)
    clip.write_gif(output_file, fps=20)

def main():    
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")

    text, length = load_example_input(cfg.DEMO.EXAMPLE)
    task = "Stylized Text2Motion"

    
    # loading checkpoints
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]

    from collections import OrderedDict
    mld_state_dict = torch.load(cfg.TRAIN.PRETRAINED_MLD,
                            map_location="cpu")["state_dict"]
    
    vae_state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE,
                                map_location="cpu")["state_dict"]
        
    # load dataset to extract nfeats dim of model
    # cuda options
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]
    model = get_model(cfg, dataset)
    model.load_state_dict(state_dict, strict=False)

    style_dict = torch.load(cfg.TRAIN.PRETRAINED_STYLE)
    model.style_function.load_state_dict(style_dict, strict=True)
    
    style_dict = torch.load("./experiments/style_encoder.pt")
    style_class = StyleClassification(nclasses=100)#.cuda()
    style_class.load_state_dict(style_dict, strict=True)
    
    dict_path = "./datasets/100STYLE_name_dict.txt"
    label_to_motion = build_dict_from_txt(dict_path)
    
    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    mld_time = time.time()

    motion_path = "./test_motion"
    motion_list = os.listdir(motion_path)

    mean = torch.tensor(dataset.hparams.mean).cuda()
    std = torch.tensor(dataset.hparams.std).cuda()

    for motion_file in motion_list:
        full_name = motion_path + "/" + motion_file
        base_name = os.path.basename(full_name).split("_")[0]

        reference_motions = np.load(full_name)
        
        m_length,_ = reference_motions.shape
        if m_length < 196:
            reference_motions = np.concatenate([reference_motions,
                                     np.zeros((196 - m_length, reference_motions.shape[1]))
                                     ], axis=0)
        
        reference_motions = torch.from_numpy(reference_motions).cuda().double()
        reference_motions = reference_motions.unsqueeze(0)


        output_dir = Path(os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),"samples_" + cfg.TIME))
        output_dir.mkdir(parents=True, exist_ok=True)

        reference_motions = (reference_motions - mean) / std
        # create mld model
        total_time = time.time()

        # ToDo
        # 1 choose task, input motion reference, text, lengths
        # 2 print task, input, output path
        #
        if not text:
            logger.info(f"Begin specific task{task}")
        
        # sample
        with torch.no_grad():
            rep_lst = []    
            rep_ref_lst = []
            texts_lst = []
            # task: input or Example
            if text:
                # prepare batch data
                batch = {"length": length, "text": text,"motion":reference_motions}

                for rep in range(cfg.DEMO.REPLICATION):
                    # text motion transfer
                    joints, feats = model(batch,feature="True")
                    
                    predict_label = []
                    for data in feats:
                        logits = style_class(data.unsqueeze(0))
                        probabilities = F.softmax(logits, dim=1)

                        predicted = torch.argmax(probabilities).item()
                        motion_name = label_to_motion[str(predicted)]
                        predict_label.append(motion_name)
                    # cal inference time

                    infer_time = time.time() - mld_time
                    num_batch = 1
                    num_all_frame = sum(batch["length"])
                    num_ave_frame = sum(batch["length"]) / len(batch["length"])

                    nsample = len(joints)
                    id = 0

                    for i in range(nsample):
                        npypath = str(output_dir /
                                    f"{base_name}_{length[i]}_batch{id}_{rep}.npy")
    
                        np.save(npypath, joints[i].detach().cpu().numpy())
                        logger.info(f"Motions are generated here:\n{npypath}")

                        fig_path = Path(str(npypath).replace(".npy",".mp4"))
                        gif_path = Path(str(npypath).replace(".npy",".gif"))
                        plot_3d_motion(fig_path,t2m_kinematic_chain, joints[i].detach().cpu().numpy(), title=batch["text"][i] + " " + predict_label[i],dataset='humanml',fps=20)
                        convert_mp4_to_gif(str(fig_path),str(gif_path))
                        
        # ToDo fix time counting
        total_time = time.time() - total_time
        print(f'SMooDi Infer time - This/Ave batch: {infer_time/num_batch:.2f}')
        print(f'SMooDi Infer FPS - Total batch: {num_all_frame/infer_time:.2f}')
        print(f'SMooDi Infer time - This/Ave batch: {infer_time/num_batch:.2f}')
        print(f'SMooDi Infer FPS - Total batch: {num_all_frame/infer_time:.2f}')
        print(
            f'SMooDi Infer FPS - Running Poses Per Second: {num_ave_frame*infer_time/num_batch:.2f}')
        print(
            f'SMooDi Infer FPS - {num_all_frame/infer_time:.2f}s')
        print(
            f'SMooDi Infer FPS - Running Poses Per Second: {num_ave_frame*infer_time/num_batch:.2f}')
        print(
            f'SMooDi Infer FPS - time for 100 Poses: {infer_time/(num_batch*num_ave_frame)*100:.2f}'
        )
        print(
            f'Total time spent: {total_time:.2f} seconds (including model loading time and exporting time).'
        )

if __name__ == "__main__":
    main()