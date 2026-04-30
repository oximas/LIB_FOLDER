
import os
import numpy as np
import torch
from lib.train.dataset.base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
import json


def _load_anno(path):
    boxes = []
    with open(path) as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 4:
                continue
            try:
                x, y, w, h = map(float, parts[:4])
                boxes.append([x, y, w, h])
            except:
                continue
    return np.array(boxes, dtype=np.float32) if boxes else None


class AIC4UAV(BaseVideoDataset):
    """
    AIC-4 UAV dataset wrapper for UETrack training pipeline.
    Reads pre-extracted frames from FRAMES_ROOT and annotations
    from the original dataset folders.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader,
                 split='train', seq_ids=None, data_root=None,
                 frames_root=None):
        super().__init__('AIC4UAV', root, image_loader)

        self.frames_root = frames_root
        self.data_root   = data_root
        self.split       = split

        # seq_ids: list of folder names under frames_root
        if seq_ids is not None:
            self.seq_list = seq_ids
        else:
            self.seq_list = sorted(os.listdir(frames_root))

        # Build sequence metadata
        self.sequence_list = self._build_seq_list()

    def _find_anno(self, folder_name):
        parts    = folder_name.split('_', 1)
        ds_name  = parts[0]
        seq_name = parts[1] if len(parts) > 1 else folder_name
        anno_dir = os.path.join(self.data_root, ds_name, seq_name)
        for fname in ['annotation.txt', 'groundtruth.txt', 'groundtruth_rect.txt']:
            p = os.path.join(anno_dir, fname)
            anno = _load_anno(p)
            if anno is not None:
                return anno
        return None

    def _build_seq_list(self):
        seqs = []
        for folder in self.seq_list:
            seq_dir = os.path.join(self.frames_root, folder)
            if not os.path.isdir(seq_dir):
                continue
            frames = sorted([f for f in os.listdir(seq_dir)
                             if f.endswith(('.jpg', '.png', '.jpeg'))])
            if not frames:
                continue
            anno = self._find_anno(folder)
            if anno is None:
                continue
            n = min(len(frames), len(anno))
            if n < 10:
                continue
            seqs.append({
                'name'   : folder,
                'seq_dir': seq_dir,
                'frames' : frames[:n],
                'anno'   : anno[:n],
            })
        print(f'AIC4UAV [{self.split}]: {len(seqs)} sequences loaded')
        return seqs

    def get_name(self):
        return 'aic4uav'

    def has_class_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info(self, seq_id):
        seq  = self.sequence_list[seq_id]
        anno = torch.from_numpy(seq['anno'])   # (N, 4) xywh
        valid = (anno[:, 2] > 0) & (anno[:, 3] > 0)
        visible = valid
        return {'bbox': anno, 'valid': valid, 'visible': visible}

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq = self.sequence_list[seq_id]
        frame_list = []
        for fid in frame_ids:
            img_path = os.path.join(seq['seq_dir'], seq['frames'][fid])
            img = self.image_loader(img_path)
            frame_list.append(img)
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        anno_frames = {k: v[frame_ids, ...] if v.ndim > 1
                       else v[frame_ids] for k, v in anno.items()}
        obj_meta = {'object_class_name': 'object',
                    'motion_class': None,
                    'major_class': None,
                    'root_class': None,
                    'motion_adverb': None}
        return frame_list, anno_frames, obj_meta
