from lhotse import CutSet, load_manifest_lazy, MonoCut
from lhotse.utils import compute_num_frames

import multiprocessing.pool as mpp
from glob import glob
from tqdm.auto import tqdm

import h5py
from math import isclose
from tpe import TPE

N_T = 8

s5_ds_dr = '/home/qanpan.lo/ds1/manifests/'

def extrat_feats(cut, h5):
    if cut.has_features:
        left_offset_frames, right_offset_frames = 0, 0
        start = cut.start
        duration = cut.duration

        if not isclose(start, cut.features.start):
            left_offset_frames = compute_num_frames(
                start - cut.features.start,
                frame_shift=cut.features.frame_shift,
                sampling_rate=cut.features.sampling_rate,
            )

        right_offset_frames = left_offset_frames + compute_num_frames(
            duration, frame_shift=cut.features.frame_shift, sampling_rate=cut.features.sampling_rate
        )

        audio_feature = h5[cut.features.storage_key][left_offset_frames:right_offset_frames].copy()

        return audio_feature
        
    return None

def analyse(tpe, cs_file):
    cs = load_manifest_lazy(cs_file)
    token_freq_map = {}

    if isinstance(cs, MonoCut):
        cs = [cs]

    features_storage_path = f'{s5_ds_dr}/{cs[0].features.storage_path.split("/")[-1]}'
    with h5py.File(features_storage_path, 'r') as h5:
        for c in cs:
            feats = extrat_feats(c, h5)
            if feats is not None:
                feats = feats[:, 0]
                st = tpe.at2st(feats)

                for token in st:
                    if token not in token_freq_map:
                        token_freq_map[token] = 0
                    token_freq_map[token] += 1

    return token_freq_map

def analyse_worker(args):
    id, cutset_files, vocab_file = args
    token_freq_map = {}
    tpe = TPE("/home/qanpan.lo/ds1/TPE/libtpe.so", vocab_file)

    if id == 0:
        print(f'Processed {id}/{len(cutset_files)} cutset files')
        for cutset_file in tqdm(cutset_files, desc="Analyse cutset files"):
            token_freq_map.update(analyse(tpe, cutset_file))
    else:
        for cutset_file in cutset_files:
            token_freq_map.update(analyse(tpe, cutset_file))

    return token_freq_map


if __name__ == '__main__':

    vocab_file = "/home/qanpan.lo/52271.txt"
    print('Loading vocab file...')
    vocab_orig = {}
    with open("/home/qanpan.lo/52271.txt", 'r') as f:
        for line in f:
            tokens = line.strip().split(maxsplit=1)
            assert len(tokens) == 2
            vocab_orig[int(tokens[0])] = tokens[1]

    cutset_files = glob(f'{s5_ds_dr}/*_cuts_valid.jsonl.gz')
    print(f'Found {len(cutset_files)} cutset files')

    cutset_file_workers = [[] for _ in range(N_T)]
    for i, cutset_file in enumerate(cutset_files):
        cutset_file_worker = []
        cutset_file_workers[i % N_T].append(cutset_file)

    for i in range(N_T):
        cutset_file_workers[i] = (i, cutset_file_workers[i], vocab_file)
    with mpp.Pool(N_T) as p:
        vocab_map = list(tqdm(p.imap_unordered(analyse_worker, cutset_file_workers), total=len(cutset_file_workers), desc='Analyse workers'))

    print("Merging token frequency maps")
    token_freq_map = {}
    for vocab in vocab_map:
        for token in vocab:
            if token not in token_freq_map:
                token_freq_map[token] = 0
            token_freq_map[token] += vocab[token]

    new_vocab = {}
    for token in token_freq_map:
        if token_freq_map[token] >= 0:
            new_vocab[token] = vocab_orig[token]

    print("original vocab size:", len(vocab_orig))
    print("new vocab size:", len(new_vocab))

    new_vocab = {k: v for k, v in sorted(new_vocab.items(), key=lambda item: item[0])}

    with open(f"{vocab_file}.new", 'w') as f:
        for token in new_vocab:
            f.write(f"{token} {new_vocab[token]}\n")


