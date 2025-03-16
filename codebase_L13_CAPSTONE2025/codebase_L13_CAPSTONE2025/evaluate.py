# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import pickle
import argparse
import yaml
import time
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
import os
import glob
import torch
import soundfile as sf
import torch.nn as nn
from utils import demix_track, demix_track_demucs, get_model_from_config
import pandas as pd
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")


drumkit_names = (
    'brooklyn',
    'east_bay',
    'heavy',
    'portland',
    'retro_rock',
    'socal',
    'bluebird',
    'detroit_garage',
    'motown_revisited',
    'roots'
)

stem_composition = {
    'kick': ['kick'],
    'snare': ['snare'],
    'toms': ['hi_tom', 'mid_tom', 'low_tom'],
    'hihat': ['hihat_closed', 'hihat_open'],
    'cymbals': ['ride', 'crash_left']
}


def nsdr(y_pred, y_true, eps: float = 1e-7, reduction: str = 'mean'):
    assert reduction in ('none', 'mean', 'sum'), f'Reduction mode \"{reduction}\" not recognized.'

    signal = torch.square(y_true).mean(dim=[-2, -1])
    distortion = torch.square(y_true - y_pred).mean(dim=[-2, -1])

    if torch.all(signal < torch.finfo(torch.float32).eps):
        signal = torch.zeros_like(signal)

    if torch.all(distortion < torch.finfo(torch.float32).eps):
        distortion = torch.zeros_like(distortion)

    ratio = 10 * torch.log10((signal + eps) / (distortion + eps))

    if reduction == 'mean':
        return ratio.mean(0)
    elif reduction == 'sum':
        return ratio.sum(0)
    elif reduction == 'none':
        return ratio


def run_folder(model, args, config, device, verbose=False):
    start_time = time.time()
    model.eval()

    data_root = '/nas/home/amezza/StemGMD/StemGMD/'
    csv = 'datasets/test_eval_session_presence.csv'
    df = pd.read_csv(csv)
    print('Total tracks found: {}'.format(len(df)))

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if not os.path.isdir(args.store_dir):
        os.mkdir(args.store_dir)

    store_dir = Path(args.store_dir)

    sdr_dict = {}
    for kit in drumkit_names:
        sdr_dict[kit] = {s: [] for s in instruments}

    running_sdr, denominator = 0, 0
    for idx, row in df.iterrows():
        path = os.path.join(data_root, row['basepath'], 'mixture', row['filename'])
        mix, sr = sf.read(path)
        mixture = torch.tensor(mix.T, dtype=torch.float32)

        with torch.no_grad():
            if args.model_type == 'htdemucs':
                res = demix_track_demucs(config, model, mixture, device)
            else:
                res = demix_track(config, model, mixture, device)

        for instr in instruments:
            indicator = bool(sum([row[df.columns.get_loc(ii)] for ii in stem_composition[instr]]))
            drumkit = Path(row['basepath']).stem

            if indicator:
                path = os.path.join(data_root, row['basepath'], instr, row['filename'])
                ref, sr = sf.read(path)
                reference = torch.tensor(ref.T, dtype=torch.float32)
                pred = torch.tensor(res[instr], dtype=torch.float32)

                sdr_value = nsdr(y_true=reference, y_pred=pred).cpu().numpy()

                if sdr_value != 0.0:
                    running_sdr += sdr_value
                    denominator += 1
                    print(f'{idx} - {instr} ({drumkit}): {sdr_value:.5f} dB (avg: {running_sdr/denominator:.5f} dB)')
                    sdr_dict[drumkit][instr].append(sdr_value)

    stop_time = time.time()

    pickle_name = f'htdemucs_nsdr_present_{Path(csv).stem}'

    if config.training.target_instrument is not None:
        pickle_name += f'_fine-tuning_{config.training.target_instrument}'

    with open(store_dir.joinpath(f'{pickle_name}.pkl'), 'wb') as f:
        pickle.dump(sdr_dict, f)

    for stem in instruments:
        tmp = []
        for kit in sdr_dict.keys():
            tmp += sdr_dict[kit][stem]
        avg_sdr = np.mean(tmp)
        print(f'- Average {stem}: {avg_sdr} dB')

    print("Elapsed time: {:.2f} sec".format(stop_time - start_time))


def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c', help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit")
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to valid weights")
    parser.add_argument("--store_dir", default="", type=str, help="path to store results as wav file")
    parser.add_argument("--device_ids", nargs='+', type=int, default=0, help='list of gpu ids')
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    torch.backends.cudnn.benchmark = True

    with open(args.config_path) as f:
        if args.model_type == 'htdemucs':
            config = OmegaConf.load(args.config_path)
        else:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    print("Instruments: {}".format(config.training.instruments))

    model = get_model_from_config(args.model_type, config)
    if args.start_check_point != '':
        print('Start from checkpoint: {}'.format(args.start_check_point))
        model.load_state_dict(
            torch.load(args.start_check_point, map_location=torch.device('cpu'))
        )

    if torch.cuda.is_available():
        device_ids = args.device_ids
        if type(device_ids) == int:
            device = torch.device(f'cuda:{device_ids}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        print('CUDA is not avilable. Run inference on CPU. It will be very slow...')
        model = model.to(device)

    run_folder(model, args, config, device, verbose=False)


if __name__ == "__main__":
    proc_folder(None)
