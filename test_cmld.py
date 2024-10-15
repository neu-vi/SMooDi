import json
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf

from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
from torch.utils.data import Dataset, DataLoader


def print_table(title, metrics):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER
    # create logger
    logger = create_logger(cfg, phase="test")
    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(OmegaConf.to_yaml(cfg))

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
        #     str(x) for x in cfg.DEVICE)
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = get_datasets(cfg, logger=logger, phase="test")[0]
    data_loader = datasets.test_dataloader() #DataLoader(datasets.Dataset, batch_size=1, shuffle=True)

    logger.info("datasets module {} initialized".format("".join(
        cfg.TRAIN.DATASETS)))

    # create model
    model = get_model(cfg, datasets)
    logger.info("model {} loaded".format(cfg.model.model_type))
    

    # optimizer
    metric_monitor = {
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
    }

    # callbacks
    callbacks = [
        pl.callbacks.RichProgressBar(),
        ProgressLogger(metric_monitor=metric_monitor),
    ]
    logger.info("Callbacks initialized")


    # loading state dict
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))

    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]
    model_state_dict = model.state_dict()

    # 过滤出匹配的权重
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].size() == v.size()}
    model.load_state_dict(filtered_state_dict,strict=False)
    state_dict = torch.load(cfg.TRAIN.PRETRAINED_MLD,
                                map_location="cpu")["state_dict"]
    # extract encoder/decoder
    from collections import OrderedDict
    denoiser_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split(".")[0] == "denoiser":
            name = k.replace("denoiser.", "")
            denoiser_dict[name] = v

    
    model.denoiser.mld_denoiser.load_state_dict(denoiser_dict, strict=True)
    style_dict = torch.load(cfg.TRAIN.PRETRAINED_STYLE)
    model.style_function.load_state_dict(style_dict, strict=True)


    model = model.cuda()
    model.eval()

    all_metrics = {}
    replication_times = cfg.TEST.REPLICATION_TIMES
    # calculate metrics
    for i in range(replication_times):
        metrics_type = ", ".join(cfg.METRIC.TYPE)
        logger.info(f"Evaluating {metrics_type} - Replication {i}")
      
        for batch_idx, batch in enumerate(data_loader):
            # model.allsplit_step_test("test", batch, batch_idx)    
            model.allsplit_step("test", batch, batch_idx)

        # metrics = model.allsplit_epoch_end_test("test")
        metrics = model.allsplit_epoch_end("test")
              
        for key, item in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = [item]
            else:
                all_metrics[key] += [item]

    all_metrics_new = {}
    for key, item in all_metrics.items():
        mean, conf_interval = get_metric_statistics(np.array(item),
                                                    replication_times)
        all_metrics_new[key + "/mean"] = mean
        all_metrics_new[key + "/conf_interval"] = conf_interval
    print_table(f"Mean Metrics", all_metrics_new)
    all_metrics_new.update(all_metrics)
    # save metrics to file
    metric_file = output_dir.parent / f"metrics_{cfg.TIME}.json"
    with open(metric_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics_new, f, indent=4)
    logger.info(f"Testing done, the metrics are saved to {str(metric_file)}")


if __name__ == "__main__":
    main()
