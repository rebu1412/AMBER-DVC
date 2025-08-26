from collections import defaultdict
from itertools import chain
import json

import os

import scipy
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.interpolate import interp1d

def collate_fn(batch):
    # A list of samples
    batch = [b for b in batch if b is not None]
    batch_size = len(batch)
    if batch_size == 0:
        return None  # hoặc raise StopIteration
    feature_size = batch[0][0].shape[1]
    feature_list, gt_timestamps_list, labels, caption_list, gt_raw_timestamp, raw_duration, raw_caption, key = zip(
        *batch)

    # Compute Maximum Lengths
    max_video_length = max([x.shape[0] for x in feature_list]) # frame dimension
    max_caption_length = max(chain(*[[len(caption) for caption in captions] for captions in caption_list]))
    total_caption_num = sum(chain([len(captions) for captions in caption_list]))

    gt_timestamps = list(chain(*gt_timestamps_list))

    # Initializing Tensors
    video_tensor = torch.FloatTensor(batch_size, max_video_length, feature_size).zero_()
    video_length = torch.FloatTensor(batch_size, 3).zero_()  # frame count, duration(s), and number of events
    video_mask = torch.BoolTensor(batch_size, max_video_length).zero_() # True for valid frames, False for padding

    caption_tensor = torch.LongTensor(total_caption_num, max_caption_length).zero_() # stores padded tokenized captions
    caption_length = torch.LongTensor(total_caption_num).zero_() # stores the length of each caption
    caption_mask = torch.BoolTensor(total_caption_num, max_caption_length).zero_() # boolean mask for valid caption tokens

    caption_gather_idx = torch.LongTensor(total_caption_num).zero_() # maps each caption to its video index in the batch
    max_caption_num = max(len(x) for x in caption_list)
    gt_boxes_tensor = torch.zeros(batch_size, max_caption_num, 2) # stores normalized event timestamps (center and width)

    total_caption_idx = 0
    total_proposal_idx = 0

    for idx in range(batch_size):
        # Video Data
        video_len = feature_list[idx].shape[0] # frame number
        video_tensor[idx, :video_len, :] = torch.from_numpy(feature_list[idx])
        
        gt_proposal_length = len(gt_timestamps_list[idx])
        video_length[idx, 0] = float(video_len) # frame count
        video_length[idx, 1] = raw_duration[idx]
        video_length[idx, 2] = gt_proposal_length # number of events

        video_mask[idx, :video_len] = True

        # Caption Mapping
        caption_gather_idx[total_caption_idx:total_caption_idx + gt_proposal_length] = idx

        # Ground-Truth Boxes
        gt_boxes_tensor[idx, :gt_proposal_length] = torch.tensor( # Normalizes timestamps to [center, l] format
            [[(ts[1] + ts[0]) / (2 * raw_duration[idx]), (ts[1] - ts[0]) / raw_duration[idx]] for ts in gt_raw_timestamp[idx]]).float()

        # Captions
        for iidx, captioning in enumerate(caption_list[idx]):
            _caption_len = len(captioning)
            caption_tensor[total_caption_idx + iidx, :_caption_len] = torch.from_numpy(captioning)
            caption_length[total_caption_idx + iidx] = _caption_len
            caption_mask[total_caption_idx + iidx, :_caption_len] = True
        total_caption_idx += gt_proposal_length
    
    # Ground-Truth Box Mask
    gt_boxes_mask = (gt_boxes_tensor != 0).sum(2) > 0 # [batch_size, max_caption_num]

    target = [{'boxes': torch.tensor([[(ts[1] + ts[0]) / (2 * raw_duration[i]), (ts[1] - ts[0]) / raw_duration[i]] for ts in gt_raw_timestamp[i]]).float(), # Normalizes [center, length].
               'labels': torch.tensor(labels[i]).long(),
               'masks': None,
               'image_id': vid} 
               for i, vid in enumerate(list(key))]

    dt = {
        "video":
            {
                "tensor": video_tensor,  # Video feature (tensor):  [bs, max_video_length, feature_size]
                "length": video_length,  # Video data    (tensor):  [bs, (frame count, duration(s), and number of events)]
                "mask": video_mask,      # Video mask    (tensor):  [bs, max_video_length]
                "key": list(key),        # Video ID      (List of dictionaries):  [bs]
                "target": target,
            },

        "gt":
            {
                "featstamps": gt_timestamps,          # list,        (gt_all_event_num, 2)
                "timestamp": list(gt_raw_timestamp),  # Raw timestamps(s)            (list): [bs, gt_event_num, 2]
                "gather_idx": caption_gather_idx,     # Caption-to-video mapping   (tensor): [gt_all_event_num]
                "boxes": gt_boxes_tensor,             # Normalized event boxes     (tensor): [bs, max_caption_num, 2]
                "boxes_mask": gt_boxes_mask,          ## Event mask                 (tensor): [bs, max_caption_num]
            },

        "cap":
            {
                "tensor": caption_tensor,  ## Tokenized captions (tensor): [gt_all_event_num, max_caption_length]
                "length": caption_length,  # Caption lengths    (tensor): [gt_all_event_num]
                "mask": caption_mask,      # Caption mask       (tensor): [gt_all_event_num, max_caption_length]
                "raw": list(raw_caption),  # Raw captions         (list): [bs, gt_event_num, ~~caption_len]
            }
    }
    dt = {k1 + '_' + k2: v2 for k1, v1 in dt.items() for k2, v2 in v1.items()}
    return dt


class Translator(object):
    def __init__(self, translator_json, vocob_size):
        self.vocab_size = vocob_size
        self.vocab = json.load(open(translator_json, 'r'))
        assert self.vocab_size == len(self.vocab['word_to_ix'].keys())
        self.vocab['word_to_ix'] = defaultdict(lambda: self.vocab_size,
                                               self.vocab['word_to_ix'])
        self.vocab['ix_to_word'] = defaultdict(lambda: self.vocab_size,
                                               self.vocab['ix_to_word'])

    def translate(self, sentence, max_len):
        tokens = [',', ':', '!', '_', ';', '-', '.', '?', '/', '"', '\\n', '\\', '.']
        for token in tokens:
            sentence = sentence.replace(token, ' ')
        sentence_split = sentence.replace('.', ' . ').replace(',', ' , ').lower().split()
        res = np.array(
            [0] + [self.vocab['word_to_ix'][word] for word in sentence_split][:max_len - 2] + [0])
        return res

    def rtranslate(self, sent_ids):
        for i in range(len(sent_ids)):
            if sent_ids[i] == 0:
                sent_ids = sent_ids[:i]
                break
        if len(sent_ids):
            return ' '.join([self.vocab['ix_to_word'][str(idx)] for idx in sent_ids]) + '.'
        else:
            return ''


class EDVCdataset(Dataset):

    def __init__(self, anno_file, feature_folder, translator_json, is_training, proposal_type, opt):

        super(EDVCdataset, self).__init__()
        self.anno = json.load(open(anno_file, 'r'))
        self.translator = Translator(translator_json, opt.vocab_size)
        self.max_caption_len = opt.max_caption_len
        self.keys = list(self.anno.keys()) # GET FILE NAME
        for json_path in opt.invalid_video_json:
            invalid_videos = json.load(open(json_path))
            self.keys = [k for k in self.keys if k[:13] not in invalid_videos]

        self.feature_folder = feature_folder
        self.feature_sample_rate = opt.feature_sample_rate
        self.opt = opt
        self.proposal_type = proposal_type
        self.is_training = is_training
        self.train_proposal_sample_num = opt.train_proposal_sample_num
        self.gt_proposal_sample_num = opt.gt_proposal_sample_num
        self.feature_dim = self.opt.feature_dim
        self.num_queries = opt.num_queries
        if 'clip' in opt.visual_feature_type:
            self.viz_feats=torch.load(feature_folder[0])

    def __len__(self):
        return len(self.keys)

    def process_time_step(self, duration, timestamps_list, feature_length):
        duration = np.array(duration)
        timestamps = np.array(timestamps_list)
        feature_length = np.array(feature_length)
        featstamps = feature_length * timestamps / duration
        featstamps = np.minimum(featstamps, feature_length - 1).astype('int')
        featstamps = np.maximum(featstamps, 0).astype('int')
        return featstamps.tolist()

    def __getitem__(self, idx):
        raise NotImplementedError()


class PropSeqDataset(EDVCdataset):

    def __init__(self, anno_file, feature_folder, translator_pickle, is_training, proposal_type, opt):
        super(PropSeqDataset, self).__init__(anno_file, feature_folder, translator_pickle, is_training, proposal_type, opt)

    def load_feats(self, key):
        vf_types = self.opt.visual_feature_type # ['resnet', 'bn']
        rescale_method = 'fix_length'
        if type(vf_types) == list:
            assert type(self.feature_folder) == list and len(vf_types) == len(self.feature_folder)
            feats_dict = {}
            all_padding = True

            # Load and process each feature type
            for vf_type, vf_folder in zip(vf_types, self.feature_folder):
                feats, is_padding = get_feats(key, vf_type, vf_folder)
                all_padding = is_padding & all_padding
                feats_dict[vf_type] = feats # [frames, dim]

                if self.opt.data_rescale:
                    if rescale_method == 'fix_length':
                        rescale_len = self.opt.frame_embedding_num
                    elif rescale_method.startswith('follow'): # follow_resnet
                        follow_type = rescale_method.split('_')[1] 
                        assert follow_type in vf_types
                        rescale_len = len(feats_dict[follow_type])
                    else:
                        raise AssertionError('rescale_method must be \"fix_length\" or "follow_*"')
                    if feats.shape[0] != rescale_len:
                        feats = resizeFeature(feats, rescale_len, 'nearest')
                else:
                    feats = feats[::self.opt.feature_sample_rate]
                feats_dict[vf_type] = feats

            if all_padding:
                print(f'all feature files of video {key} do not exist')
            out = np.concatenate([feats_dict[type_] for type_ in vf_types], axis=-1)
        else:
            out, is_padding = get_feats(key, vf_types, self.feature_folder, data_norm=self.opt.data_norm)
            if self.opt.data_rescale:
                out = resizeFeature(out, self.opt.frame_embedding_num, 'nearest')
        assert out.shape[1] == self.feature_dim, 'wrong value of feature_dim'
        return out

    def __getitem__(self, idx):
        key = str(self.keys[idx])
        if 'clip' in self.opt.visual_feature_type:
            try:
                feats = self.viz_feats[key].numpy()
                vf_types = self.opt.visual_feature_type # ['resnet', 'bn']
                rescale_method = 'fix_length'
                if type(vf_types) == list:
                    assert type(self.feature_folder) == list and len(vf_types) == len(self.feature_folder)
                    feats_dict = {}
                    all_padding = True

                    # Load and process each feature type
                    for vf_type, vf_folder in zip(vf_types, self.feature_folder):
                        feats, is_padding = get_feats(key, vf_type, vf_folder)
                        all_padding = is_padding & all_padding
                        feats_dict[vf_type] = feats # [frames, dim]

                        if self.opt.data_rescale:
                            if rescale_method == 'fix_length':
                                rescale_len = self.opt.frame_embedding_num
                            elif rescale_method.startswith('follow'): # follow_resnet
                                follow_type = rescale_method.split('_')[1] 
                                assert follow_type in vf_types
                                rescale_len = len(feats_dict[follow_type])
                            else:
                                raise AssertionError('rescale_method must be \"fix_length\" or "follow_*"')
                            if feats.shape[0] != rescale_len:
                                feats = resizeFeature(feats, rescale_len, 'nearest')
                        else:
                            feats = feats[::self.opt.feature_sample_rate]
                        feats_dict[vf_type] = feats
            except:
                print('{} not exists, use zero padding. '.format(key))
                feats = np.zeros((200, 768))
                # return None
        else:
            feats = self.load_feats(key)
        duration = self.anno[key]['duration']
        captions = self.anno[key]['sentences']
        gt_timestamps = self.anno[key]['timestamps']  # [gt_num, 2]
        action_labels = self.anno[key].get('action_labels', [0] * len(gt_timestamps))
        assert max(action_labels) <= self.opt.num_classes

        # Sampling
        gt_sample_num = len(gt_timestamps) if (len(gt_timestamps) < self.gt_proposal_sample_num) \
                                           else self.gt_proposal_sample_num
        random_ids = np.random.choice(list(range(len(gt_timestamps))), gt_sample_num, replace=False)
        captions = [captions[_] for _ in range(len(captions)) if _ in random_ids]
        gt_timestamps = [gt_timestamps[_] for _ in range(len(gt_timestamps)) if _ in random_ids]
        action_labels = [action_labels[_] for _ in range(len(action_labels)) if _ in random_ids]

        caption_label = [np.array(self.translator.translate(sent, self.max_caption_len)) for sent in captions]
        gt_featstamps = self.process_time_step(duration, gt_timestamps, feats.shape[0])

        # lnt_timestamps = gt_timestamps
        # lnt_featstamps = gt_featstamps
        # gt_idx = np.arange(len(gt_timestamps))
        # event_seq_idx = seq_gt_idx = np.expand_dims(gt_idx, 0)
        # lnt_scores = [1.] * len(lnt_featstamps)

        return feats, gt_featstamps, action_labels, caption_label, gt_timestamps, duration, captions, key
        # feats: [frame, dim]
        # gt_featstamps: [events, 2] (f)       # action_labels: [events]
        # caption_label: [events, L]
        # gt_timestamps: [events, 2] (s)
        # duration: float
        # captions: ["spread margarine...", "place a slice..."]
        # key: "v_GLd3aX16zBg" (example)


def iou(interval_1, interval_2):
    interval_1, interval_2 = map(np.array, (interval_1, interval_2))
    start, end = interval_2[None, :, 0], interval_2[None, :, 1]
    start_i, end_i = interval_1[:, None, 0], interval_1[:, None, 1]
    intersection = np.minimum(end, end_i) - np.maximum(start, start_i)
    union = np.minimum(np.maximum(end, end_i) - np.minimum(start, start_i), end - start + end_i - start_i)
    iou = intersection.clip(0) / (union + 1e-8)
    return iou


def sort_events(proposal_data):
    for vid in proposal_data.keys():
        v_data = proposal_data[vid]
        v_data = [p for p in v_data if p['score'] > 0]
        tmp = sorted(v_data, key=lambda x: x['segment'])
        proposal_data[vid] = tmp
    return proposal_data


def read_file(path, feat_dim, MEAN=0., VAR=1., data_norm=False):
    if os.path.exists(path) or 's3://' in path:
        ext = path.split('.')[-1]
        if ext == 'npy':
            feats = np.load(path)
        elif ext == 'csv':
            feats = pd.read_csv(path).values
        elif ext == 'pkl':
            feats = pickle.load(open(path))
        else:
            raise NotImplementedError
        padding = False
    else:
        print('{} not exists, use zero padding. '.format(path))
        feats = np.zeros((100, feat_dim))
        padding = True
    if data_norm:
        feats = (feats - MEAN) / np.sqrt(VAR)
    return feats, padding


def get_feats(key, vf_type, vf_folder, data_norm=False):
    MEAN = VAR = 0

    if vf_type == 'c3d':
        feat_dim = 500
        MEAN = -0.001915027447565527
        VAR = 1.9239444588254049
        path = os.path.join(vf_folder, key[0:13] + '.npy')

    elif vf_type == 'resnet':
        feat_dim = 2048
        MEAN = 0.41634243404998694
        VAR = 0.2569392081183313
        path = os.path.join(vf_folder, key[2:13] + '_resnet.npy')

    elif vf_type == 'bn':
        feat_dim = 1024
        MEAN = 0.8945046635916155
        VAR = 3.6579982046018844
        path = os.path.join(vf_folder, key[2:13] + '_bn.npy')

    elif vf_type == 'tsn_100':
        feat_dim = 400
        path = os.path.join(vf_folder, key[0:13] + '.csv')

    elif vf_type == 'i3d_rgb':
        feat_dim = 1024
        path = os.path.join(vf_folder, key[:13] + '_rgb.npy')

    elif vf_type == 'i3d_flow':
        feat_dim = 1024
        path = os.path.join(vf_folder, key[:13] + '_flow.npy')

    elif vf_type == 'tsp':
        feat_dim = 512
        path = os.path.join(vf_folder, key[0:13] + '.npy')

    elif vf_type == 'vggish':
        feat_dim = 128
        path = os.path.join(vf_folder, key[0:13] + '.npy')

    else:
        raise AssertionError('feature type error')

    feats, padding = read_file(path, feat_dim, MEAN, VAR, data_norm)

    if len(feats.shape) == 1:
        assert feats.shape[0] == feat_dim, 'load {} error, got shape {}'.format(path, feats.shape)

    assert feats.shape[1] == feat_dim, 'load {} error, got shape {}'.format(path, feats.shape)
    return feats, padding


def resizeFeature(inputData, newSize, sample_method):
    # inputX: (temporal_length,feature_dimension) #
    originalSize = len(inputData)
    # print originalSize
    if originalSize == 1:
        inputData = np.reshape(inputData, [-1])
        return np.stack([inputData] * newSize)
    x = np.array(range(originalSize))
    f = interp1d(x, inputData, axis=0, kind=sample_method)
    x_new = [i * float(originalSize - 1) / (newSize - 1) for i in range(newSize)]
    y_new = f(x_new)
    return y_new
