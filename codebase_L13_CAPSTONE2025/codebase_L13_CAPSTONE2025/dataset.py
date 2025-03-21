# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'


import os
import random
import numpy as np
import torch
import soundfile as sf
import pickle
import time
from tqdm import tqdm
from glob import glob
import audiomentations as AU
import pedalboard as PB


def load_chunk(path, length, chunk_size, offset=None):
    if chunk_size <= length:
        if offset is None:
            offset = np.random.randint(length - chunk_size + 1)
        x = sf.read(path, dtype='float32', start=offset, frames=chunk_size)[0]
    else:
        x = sf.read(path, dtype='float32')[0]
        pad = np.zeros([chunk_size - length, 2])
        x = np.concatenate([x, pad])
    return x.T


def get_transforms_simple(instr):
    if instr == 'vocals':
        augment = AU.Compose([
            AU.TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=0.1),
            AU.PitchShift(min_semitones=-4, max_semitones=4, p=0.1),
            AU.Mp3Compression(min_bitrate=32, max_bitrate=320, backend="lameenc", p=0.1),  # reduce bitrate (max kbps range: [8, 320]) with probability 0.5
        ], p=1.0)
    elif instr == 'other':
        augment = AU.Compose([
            AU.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.1),
            AU.TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=True, p=0.1),
            AU.PitchShift(min_semitones=-4, max_semitones=4, p=0.1),
            AU.Mp3Compression(min_bitrate=32, max_bitrate=320, backend="lameenc", p=0.1),  # reduce bitrate (max kbps range: [8, 320]) with probability 0.5
        ], p=1.0)
    else:
        print('Error no augms for: {}'.format(instr))
        augment = AU.Compose([], p=0.0)
    return augment


# TODO: Technically not used...
def get_transforms_drums(instr):
    if instr in ['hihat', 'cymbals']:
        augment = AU.Compose([
            AU.TanhDistortion(min_distortion=0.0, max_distortion=0.75, p=0.2),
            AU.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.1),
            AU.Mp3Compression(min_bitrate=32, max_bitrate=320, backend="lameenc", p=0.1),  # reduce bitrate (max kbps range: [8, 320]) with probability 0.5
        ], p=1.0)
    elif instr in ['kick', 'snare', 'toms']:
        augment = AU.Compose([
            AU.TanhDistortion(min_distortion=0.0, max_distortion=0.75, p=0.2),
            AU.Mp3Compression(min_bitrate=32, max_bitrate=320, backend="lameenc", p=0.1),  # reduce bitrate (max kbps range: [8, 320]) with probability 0.5
        ], p=1.0)
    else:
        print(f'Error no augms for: {instr}')
        augment = AU.Compose([], p=0.0)
    return augment


class MSSDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_path, metadata_path="metadata.pkl", dataset_type=1):
        self.config = config
        self.dataset_type = dataset_type  # 1, 2, 3 or 4
        self.instruments = instruments = config.training.instruments

        # Augmentation block
        self.aug = False
        if config.training.augmentation == 1:
            print('Use augmentation for training')
            self.aug = True
        else:
            print('Augmentation is disabled')

        self.mp3_aug = None
        if config.training.use_mp3_compress:
            self.mp3_aug = AU.Compose([
                AU.Mp3Compression(min_bitrate=32, max_bitrate=320, backend="lameenc", p=0.1),
            ], p=1.0)

        self.augment_func = dict()
        for instr in self.instruments:
            self.augment_func[instr] = None

        if self.config.training.augmentation_type == "simple1":
            print('Use simple1 set of augmentations')
            for instr in self.instruments:
                self.augment_func[instr] = get_transforms_simple(instr)
        elif self.config.training.augmentation_type == "drums1":
            print('Use drums1 set of augmentations')
            for instr in self.instruments:
                self.augment_func[instr] = get_transforms_drums(instr)

        # metadata_path = data_path + '/metadata'
        try:
            metadata = pickle.load(open(metadata_path, 'rb'))
            print(f'Loading songs data from cache: {metadata_path}. If you updated dataset remove {os.path.basename(metadata_path)} before training!')
        except Exception:
            print('Collecting metadata for', str(data_path), 'Dataset type:', self.dataset_type)
            if self.dataset_type in [1, 4]:
                metadata = []
                track_paths = []
                if type(data_path) == list:
                    for tp in data_path:
                        track_paths += sorted(glob(tp + '/*'))
                else:
                    track_paths += sorted(glob(data_path + '/*'))

                track_paths = [path for path in track_paths if os.path.basename(path)[0] != '.' and os.path.isdir(path)]
                for path in tqdm(track_paths):
                    length = len(sf.read(path + f'/{instruments[0]}.wav')[0])
                    metadata.append((path, length))
            elif self.dataset_type == 2:
                metadata = dict()
                for instr in self.instruments:
                    metadata[instr] = []
                    track_paths = []
                    if type(data_path) == list:
                        for tp in data_path:
                            track_paths += sorted(glob(tp + '/{}/*.wav'.format(instr)))
                    else:
                        track_paths += sorted(glob(data_path + '/{}/*.wav'.format(instr)))

                    for path in tqdm(track_paths):
                        length = len(sf.read(path)[0])
                        metadata[instr].append((path, length))
            elif self.dataset_type == 3:
                import pandas as pd
                if type(data_path) != list:
                    data_path = [data_path]

                metadata = dict()
                for i in range(len(data_path)):
                    print('Reading tracks from: {}'.format(data_path[i]))
                    df = pd.read_csv(data_path[i])

                    skipped = 0
                    for instr in self.instruments:
                        part = df[df['instrum'] == instr].copy()
                        print('Tracks found for {}: {}'.format(instr, len(part)))
                    for instr in self.instruments:
                        part = df[df['instrum'] == instr].copy()
                        metadata[instr] = []
                        track_paths = list(part['path'].values)
                        for path in tqdm(track_paths, desc=f'Cache {instr} metadata'):
                            if not os.path.isfile(path):
                                print('Cant find track: {}'.format(path))
                                skipped += 1
                                continue
                            # print(path)
                            try:
                                length = len(sf.read(path)[0])
                            except:
                                print('Problem with path: {}'.format(path))
                                skipped += 1
                                continue
                            metadata[instr].append((path, length))
                    if skipped > 0:
                        print('Missing tracks: {} from {}'.format(skipped, len(df)))
            else:
                print('Unknown dataset type: {}. Must be 1, 2 or 3'.format(self.dataset_type))
                exit()

            pickle.dump(metadata, open(metadata_path, 'wb'))

        if self.dataset_type in [1, 4]:
            print('Found tracks in dataset: {}'.format(len(metadata)))
        else:
            for instr in self.instruments:
                print('Found tracks for {} in dataset: {}'.format(instr, len(metadata[instr])))
        self.metadata = metadata
        self.chunk_size = config.audio.chunk_size
        self.min_mean_abs = config.audio.min_mean_abs

    def __len__(self):
        return self.config.training.num_steps * self.config.training.batch_size

    def load_source(self, metadata, instr):
        while True:
            if self.dataset_type in [1, 4]:
                track_path, track_length = random.choice(metadata)
                source = load_chunk(track_path + f'/{instr}.wav', track_length, self.chunk_size)
            else:
                track_path, track_length = random.choice(metadata[instr])
                source = load_chunk(track_path, track_length, self.chunk_size)
            if np.abs(source).mean() >= self.min_mean_abs:  # remove quiet chunks
                break
        source = self.always_augm_data(source)
        if self.aug:
            source = self.augm_data(source, instr)
        return torch.tensor(source, dtype=torch.float32)

    def load_random_mix(self):
        res = []
        for instr in self.instruments:
            # Multiple mix of sources
            s1 = self.load_source(self.metadata, instr)

            if self.config.training.augmentation_mix:
                # Remove the following if we want mixup augmentation for the target instrument
                # if instr != self.config.training.target_instrument or instr in ['toms', 'cymbals']:
                if random.uniform(0, 1) < 0.2:
                    s2 = self.load_source(self.metadata, instr)
                    w1 = random.uniform(0.5, 1.5)
                    w2 = random.uniform(0.5, 1.5)
                    s1 = (w1 * s1 + w2 * s2) / (w1 + w2)
                    if random.uniform(0, 1) < 0.1:
                        s2 = self.load_source(self.metadata, instr)
                        w1 = random.uniform(0.5, 1.5)
                        w2 = random.uniform(0.5, 1.5)
                        s1 = (w1 * s1 + w2 * s2) / (w1 + w2)

            res.append(s1)
        res = torch.stack(res)
        return res

    def load_aligned_data(self):
        track_path, track_length = random.choice(self.metadata)
        res = []
        for i in self.instruments:
            attempts = 10
            while attempts:
                source = load_chunk(track_path + f'/{i}.wav', track_length, self.chunk_size)
                if np.abs(source).mean() >= self.min_mean_abs:  # remove quiet chunks
                    break
                attempts -= 1
                if attempts <= 0:
                    print('Attempts max!', track_path)
            res.append(source)
        res = np.stack(res, axis=0)
        for i, instr in enumerate(self.instruments):
            res[i] = self.always_augm_data(res[i])
            if self.aug:
                res[i] = self.augm_data(res[i], instr)
        return torch.tensor(res, dtype=torch.float32)

    def always_augm_data(self, source):
        # Constant power pan-rule
        if random.uniform(0, 1) < 0.1:
            source = source.copy()
            mono_source = source.mean(0)
            pan = random.uniform(0, np.pi / 2)  # random panning
            source[0] = np.cos(pan) * mono_source
            source[1] = np.sin(pan) * mono_source

        # Channel shuffle
        if random.uniform(0, 1) < 0.5:
            source = source[::-1].copy()

        # Random polarity (multiply -1)
        if random.uniform(0, 1) < 0.5:
            source = -source.copy()

        return source

    def augm_data(self, source, instr):
        """source.shape = (2, 261120)"""

        # TODO: moved outside
        # # TODO: check (my) constant power pan-rule.
        # if random.uniform(0, 1) < 0.1:
        #     source = source.copy()
        #     pan = random.uniform(0, np.pi / 2)      # random panning
        #     ch = random.randint(0, 1)               # random channel
        #     source[0] = np.cos(pan) * source[ch]
        #     source[1] = np.sin(pan) * source[ch]
        #
        # # Channel shuffle
        # if random.uniform(0, 1) < 0.5:
        #     source = source[::-1].copy()

        # TODO: no reverse
        # Random inverse (do with low probability)
        # if random.uniform(0, 1) < 0.1:
        #     source = source[:, ::-1].copy()

        # TODO: moved outside
        # # Random polarity (multiply -1)
        # if random.uniform(0, 1) < 0.5:
        #     source = -source.copy()

        if self.augment_func[instr]:
            source_init = source.copy()
            source = self.augment_func[instr](samples=source, sample_rate=44100)
            if source_init.shape != source.shape:
                source = source[..., :source_init.shape[-1]]

        board_fx = []

        # TODO: no guitar-like distortion
        # Random Distortion
        # if random.uniform(0, 1) < 0.05:
        #     drive_db = random.uniform(1.0, 25.0)
        #     board_fx += [PB.Distortion(drive_db=drive_db)]

        # Random Compression
        if random.uniform(0, 1) < 0.05:
            threshold_db = random.uniform(-50.0, 0.0)
            ratio = random.uniform(1.0, 25.0)
            board_fx += [PB.Compressor(threshold_db=threshold_db, ratio=ratio)]

        # Random PitchShift
        if random.uniform(0, 1) < 0.05:
            semitones = random.uniform(-3, 3)  # Originally, (-7, 7)
            board_fx += [PB.PitchShift(semitones=semitones)]

        # TODO: finalize the instrument list
        if instr in ['hihat', 'cymbals']:
            # Random Chorus
            if random.uniform(0, 1) < 0.05:
                rate_hz = random.uniform(1.0, 7.0)
                depth = random.uniform(0.25, 0.95)
                centre_delay_ms = random.uniform(3, 10)
                feedback = random.uniform(0.0, 0.5)
                mix = random.uniform(0.1, 0.9)
                board_fx += [PB.Chorus(
                    rate_hz=rate_hz,
                    depth=depth,
                    centre_delay_ms=centre_delay_ms,
                    feedback=feedback,
                    mix=mix,
                )]

            # Random Phaser
            if random.uniform(0, 1) < 0.05:
                rate_hz = random.uniform(1.0, 10.0)
                depth = random.uniform(0.25, 0.95)
                centre_frequency_hz = random.uniform(200, 12000)
                feedback = random.uniform(0.0, 0.5)
                mix = random.uniform(0.1, 0.9)
                board_fx += [PB.Phaser(
                    rate_hz=rate_hz,
                    depth=depth,
                    centre_frequency_hz=centre_frequency_hz,
                    feedback=feedback,
                    mix=mix,
                )]

        # Random Reverb
        if random.uniform(0, 1) < 0.05:
            room_size = random.uniform(0.1, 0.9)
            damping = random.uniform(0.1, 0.9)
            wet_level = random.uniform(0.1, 0.9)
            dry_level = random.uniform(0.1, 0.9)
            width = random.uniform(0.9, 1.0)
            board_fx += [PB.Reverb(
                room_size=room_size,  # 0.1 - 0.9
                damping=damping,  # 0.1 - 0.9
                wet_level=wet_level,  # 0.1 - 0.9
                dry_level=dry_level,  # 0.1 - 0.9
                width=width,  # 0.9 - 1.0
                freeze_mode=0.0,
            )]

        # Random Resample
        if random.uniform(0, 1) < 0.05:
            target_sample_rate = random.uniform(16000, 44100)  # Originally, (4000, 44100)
            board_fx += [PB.Resample(target_sample_rate=target_sample_rate)]

        # Random Bitcrash
        if random.uniform(0, 1) < 0.05:
            bit_depth = random.uniform(4, 16)
            board_fx += [PB.Bitcrush(bit_depth=bit_depth)]

        # Random MP3Compressor
        if random.uniform(0, 1) < 0.05:
            vbr_quality = random.uniform(0, 9.999)
            board_fx += [PB.MP3Compressor(vbr_quality=vbr_quality)]

        if board_fx:
            source = PB.Pedalboard(board_fx)(source, 44100)

        return source

    def __getitem__(self, index):
        if self.dataset_type in [1, 2, 3]:
            res = self.load_random_mix()
        else:
            res = self.load_aligned_data()

        # Randomly change loudness of each stem
        if self.config.training.augmentation_loudness:
            if self.config.training.augmentation_loudness_type == 1:
                split = random.uniform(
                    self.config.training.augmentation_loudness_min,
                    self.config.training.augmentation_loudness_max
                )
                res[0] *= split
                res[1] *= (2 - split)
            else:
                for i in range(len(res)):
                    loud = random.uniform(
                        self.config.training.augmentation_loudness_min,
                        self.config.training.augmentation_loudness_max
                    )
                    res[i] *= loud

        mix = res.sum(0)

        if self.mp3_aug is not None:
            mix = self.mp3_aug(samples=mix, sample_rate=44100)

        # If we need given stem (for roformers)
        if self.config.training.target_instrument is not None:
            index = self.config.training.instruments.index(self.config.training.target_instrument)
            # if self.config.model in ['mel_band_roformer', 'bs_roformer' 'htdemucs']:
            return res[index], mix

        return res, mix
