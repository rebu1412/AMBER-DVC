# ------------------------------------------------------------------------
# PDVC
# ------------------------------------------------------------------------
# Modified from Deformable DETR(https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
import copy
import math

from misc.detr_utils import box_ops
from misc.detr_utils.misc import (inverse_sigmoid)

from .matcher import build_matcher

from .base_encoder import build_base_encoder
from .decoder import build_decoder
from .processor import build_processor
from amber.CaptioningHead import build_captioner
from .criterion import SetCriterion

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# https://github.dev/Bombomls21/DenseMambav2
# https://github.dev/Bombomls21/DenseMambav2/blob/63f19da4797f8e010c19a210f9d0432c791c10f4/baseline/pdvc/decoder_mix.py
# https://github.dev/Bombomls21/DenseMamba/tree/7895b846643662518643ff59c86c1da67bb207cb/baseline

class PDVC(nn.Module):
    def __init__(self, base_encoder, decoder, processor, captioner, 
                 num_classes, num_feature_levels,
                 aux_loss=True, with_box_refine=False, opt=None, translator=None):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            captioner: captioning head for generate a sentence for each event queries
            num_classes: number of foreground classes
            num_queries: number of event queries. This is the maximal number of events
                         PDVC can detect in a single video. For ActivityNet Captions, we recommend 10-30 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            opt: all configs
        """
        super().__init__()
        self.opt = opt
        self.base_encoder = base_encoder
        self.decoder = decoder
        self.processor = processor
        self.caption_head = captioner
        self.num_feature_levels = num_feature_levels
        self.aux_loss = aux_loss # True
        self.share_caption_head = opt.share_caption_head # 1
        self.with_box_refine = with_box_refine # 1

        hidden_dim = decoder.d_model*4

        # queries tokens
        self.num_queries = 25
        self.query_embed = nn.Embedding(self.num_queries, decoder.d_model)
        self.pos_query = nn.Parameter(torch.randn(1, decoder.d_model))

        # Head
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.count_head = nn.Linear(hidden_dim, opt.max_eseq_length + 1)
        self.bbox_head = MLP(hidden_dim, hidden_dim, 2, 3)
        # Initialization
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_head.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_head.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_head.layers[-1].bias.data, 0)

        # Auxiliary head
        num_pred = decoder.num_layers
        if self.share_caption_head:
            print('all decoder layers share the same caption head')
            self.caption_head = nn.ModuleList([self.caption_head for _ in range(num_pred)])
        else:
            print('do NOT share the caption head')
            self.caption_head = _get_clones(self.caption_head, num_pred)

        if with_box_refine:
            self.class_head = _get_clones(self.class_head, num_pred)
            self.count_head = _get_clones(self.count_head, num_pred)
            self.bbox_head = _get_clones(self.bbox_head, num_pred)
            nn.init.constant_(self.bbox_head[0].layers[-1].bias.data[1:], -2)
        else:
            nn.init.constant_(self.bbox_head.layers[-1].bias.data[1:], -2)
            self.class_head = nn.ModuleList([self.class_head for _ in range(num_pred)])
            self.count_head = nn.ModuleList([self.count_head for _ in range(num_pred)])
            self.bbox_head = nn.ModuleList([self.bbox_head for _ in range(num_pred)])
            
        self.translator = translator

        self.disable_mid_caption_heads = opt.disable_mid_caption_heads
        if self.disable_mid_caption_heads:
            print('only calculate caption loss in the last decoding layer')

    def forward(self, dt, criterion, eval_mode=False):
        vf = dt['video_tensor']  # (B, F, 3072), F = 200
        mask = ~dt['video_mask']  # (B, F)
        duration = dt['video_length'][:, 1] # (B, ), raw_duration
        bs, F, d = vf.shape
        print("[DEBUG]vf.shape", vf.shape)

        # Base Encoder
        srcs, masks, pos = self.base_encoder(vf, mask, duration)

        # Prepare Decoder input (add query tokens)
        query_embed = self.query_embed.weight
        srcs, query_indices, src_flatten, mask_flatten, lv_pos_embed_flatten = self.decoder.prepare_decoder_inputs(
            srcs, masks, pos, query_embed) # src_flatten: (B, F+F/2+F/4+F/8, hidden_dim)

        # Decoder
        hs_res, hs_res_list = self.decoder(srcs, src_flatten, lv_pos_embed_flatten) # List (each layers) of List (resolution)
        proposals_feat, anchor_boxes = self.retrieve_proposal(hs_res, query_indices)
        anchor_boxes = anchor_boxes.unsqueeze(0).repeat(bs, 1, 1)

        # Process proposal feature between different resolution
        proposals_feature = []
        for pro_res in proposals_feat:
            pro_res = pro_res.view(pro_res.shape[0], pro_res.shape[1], -1)
            proposals_feature.append(pro_res)
        proposals_feature = torch.stack(proposals_feature, dim=0) # [dec_num, bs, num_queries, dim*num_feature_levels]

        # proposals_feature = proposals_feat[-1].view(proposals_feat[-1].shape[0], proposals_feat[-1].shape[1], -1)

        # Prediction Head
        proposals = {
            'proposals_feature': proposals_feature, 
            'proposals_cl': anchor_boxes.to(proposals_feature.device)
            }
        out, loss = self.parallel_prediction(dt, proposals, criterion, eval_mode=eval_mode)

        return out, loss
    
    def retrieve_proposal(self, hs_res, query_indices):
        # Proposal feature
        proposals = []
        for hs_layer in hs_res:
            proposal = []
            for res, indices in zip(hs_layer, query_indices):
                feature = res[:, indices, :]
                proposal.append(feature)
            proposal = torch.stack(proposal, dim=2)
            proposals.append(proposal)

        # Anchor box
        frame_num = self.opt.frame_embedding_num
        group = query_indices[0][0]
        center = (torch.arange(group, frame_num, step=group) + 0.5)/frame_num
        length = torch.full_like(center, (group*2)/frame_num)
        anchor_boxes = torch.stack([center, length], dim=-1)

        return proposals, anchor_boxes

    def predict_event_num(self, counter, hs_lid):
        hs_lid_pool = torch.max(hs_lid, dim=1, keepdim=False)[0]  # [bs, feat_dim], return values
        outputs_class0 = counter(hs_lid_pool)
        return outputs_class0

    def parallel_prediction(self, dt, proposals, criterion, eval_mode=False):

        outputs_classes = []
        outputs_counts = []
        outputs_coords = []
        outputs_cap_probs = []
        outputs_cap_seqs = []

        proposals_feature = proposals['proposals_feature']
        proposals_bbox = proposals['proposals_cl']

        num_pred = proposals_feature.shape[0]
        for idx_layer, proposals_layer in enumerate(proposals_feature):
            # Predict class, count, bbox
            outputs_class = self.class_head[idx_layer](proposals_layer) # [bs, queries, 1]
            outputs_count = self.predict_event_num(self.count_head[idx_layer], proposals_layer) # [bs, opt.max_eseq_length]
            outputs_bbox = self.bbox_head[idx_layer](proposals_layer)  # [bs, queries, 2] normalize c,l

            # Caption Prediction (Pre-caption, used for evaluation)
            if eval_mode or self.opt.caption_loss_coef == 0:
                cap_probs, seq = self.caption_prediction_eval(self.caption_head[idx_layer], dt, proposals_layer, self.opt.caption_decoder_type)
            else:
                _, _, cap_probs, seq = self.caption_prediction(self.caption_head[idx_layer], dt, proposals_layer, 'none')

            # Predict bbox
            proposals_bbox = inverse_sigmoid(proposals_bbox)
            proposals_bbox = proposals_bbox + outputs_bbox
            proposals_bbox = proposals_bbox.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_counts.append(outputs_count)
            outputs_coords.append(proposals_bbox)
            outputs_cap_probs.append(cap_probs)
            outputs_cap_seqs.append(seq)

        # Stack outputs
        outputs_class = torch.stack(outputs_classes)  # [decoder_layer, bs, num_queries, 1]
        outputs_count = torch.stack(outputs_counts) # [decoder_layer, bs, opt.max_eseq_length]
        outputs_coord = torch.stack(outputs_coords)  # [decoder_layer, bs, num_queries, 2]

        all_out = {'pred_logits': outputs_class,
                   'pred_count': outputs_count,
                   'pred_boxes': outputs_coord,
                   'caption_probs': outputs_cap_probs,
                   'seq': outputs_cap_seqs}

        # Output of final layer
        out = {k: v[-1] for k, v in all_out.items()}
        
        # Loss compute
        if eval_mode or self.opt.caption_loss_coef == 0: 
            # Evaluation
            if self.aux_loss: # Auxiliary losses
                ks, vs = list(zip(*(all_out.items())))
                out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)] # list of dictionaries

            # dict_keys(['loss_ce', 'loss_counter', 'loss_bbox', 'loss_giou', 'loss_self_iou', 'cardinality_error', 
            #            'loss_ce_0', 'loss_counter_0', 'loss_bbox_0', 'loss_giou_0', 'loss_self_iou_0', 'cardinality_error_0'])
            loss, last_indices, aux_indices = criterion(out, dt['video_target'])
            
        else: 
            # Training
            if self.aux_loss: # Auxiliary losses
                ks, vs = list(zip(*(all_out.items())))
                out['aux_outputs'] = [{ks[i]: vs[i][j] for i in range(len(ks))} for j in range(num_pred - 1)] # list of dictionaries
                loss, last_indices, aux_indices = criterion(out, dt['video_target'])

                for idx_layer, proposals_layer in enumerate(proposals_feature):
                    indices = last_indices[0] if (idx_layer == num_pred - 1) else aux_indices[idx_layer][0]
                    cap_loss, cap_probs, seq = self.caption_prediction(self.caption_head[idx_layer], dt, proposals_layer,
                                                                       self.opt.caption_decoder_type, indices)

                    l_dict = {'loss_caption': cap_loss}
                    if idx_layer != num_pred - 1:
                        l_dict = {k + f'_{idx_layer}': v for k, v in l_dict.items()}
                    loss.update(l_dict)
                # dict_keys(['loss_ce', 'loss_counter', 'loss_bbox', 'loss_giou', 'loss_self_iou', 'cardinality_error', 
                #            'loss_ce_0', 'loss_counter_0', 'loss_bbox_0', 'loss_giou_0', 'loss_self_iou_0', 'cardinality_error_0', 
                #            'loss_caption_0', 'loss_caption'])
                out.update({'caption_probs': cap_probs, 'seq': seq})
            else:
                loss, last_indices = criterion(out, dt['video_target'])

                indices = last_indices[0]
                cap_loss, cap_probs, seq = self.caption_prediction(self.caption_head[idx_layer], dt, proposals_layer,
                                                                self.opt.caption_decoder_type, indices)
                
                loss.update({'loss_caption': cap_loss})

                out.pop('caption_losses', None)
                out.pop('caption_costs', None)
                out.update({'caption_probs': cap_probs, 'seq': seq})

        return out, loss
    
    def caption_prediction(self, cap_head, dt, proposals, captioner_type, indices=None):
        '''
        cap_head
        dt: ['video_tensor', 'video_length', 'video_mask', 'video_key', 'video_target', 
             'gt_featstamps', 'gt_timestamp', 'gt_gather_idx', 'gt_boxes', 'gt_boxes_mask', 
             'cap_tensor', 'cap_length', 'cap_mask', 'cap_raw']
        queries: [b, queries, dim]
        reference: [b, queries, 1]
        others: ['memory', 'mask_flatten', 'spatial_shapes', 'level_start_index', 'valid_ratios', 'proposals_mask']
        captioner_type: 
        indices=None
        '''

        N_, N_q, C = proposals.shape
        all_cap_num = len(dt['cap_tensor']) # Number of ground-truth captions in the batch
        query_mask = torch.ones(N_, proposals.shape[1], device=proposals.device).bool() # [b, queries]
        gt_mask = dt['gt_boxes_mask'] # [b, all_cap_num], boolean mask indicating valid ground-truth boxes or captions
        
        # mapping between queries and ground-truth annotations
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item()) # [total_valid_queries, total_valid_gt_boxes]
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu() # [bs], [bs]

        # Get valid features from hs based on query_mask
        hs_r = torch.masked_select(proposals, query_mask.unsqueeze(-1)).reshape(-1, C) # [total_valid_queries, hidden_dim]

        # Indices Processing
        if indices == None:
            # Evaluation Mode
            # Match all valid queries with all valid annotations in each sample of the batch
            row_idx, col_idx = 0, 0
            for i in range(N_):
                mix_mask[row_idx: (row_idx + query_nums[i]), col_idx: (col_idx + gt_nums[i])] = 1
                row_idx=row_idx + query_nums[i]
                col_idx= col_idx + gt_nums[i]

            # Match between valid queries and valid caption, many-to-many
            bigids = mix_mask.nonzero(as_tuple=False) # [num_nonzero, 2]
            feat_bigids = bigids[:, 0]  # feature indices [14688]
            cap_bigids = bigids[:, 1] # caption indices

        else:
            feat_bigids = torch.zeros(sum([len(_[0]) for _ in indices])).long()
            cap_bigids = torch.zeros_like(feat_bigids)

            total_query_ids = 0
            total_cap_ids = 0
            total_ids = 0
            max_pair_num = max([len(_[0]) for _ in indices])
            
            new_hr_for_dsa = torch.zeros(N_, max_pair_num, C)  # only for lstm-dsa
            cap_seq = dt['cap_tensor']
            new_seq_for_dsa = torch.zeros(N_, max_pair_num, cap_seq.shape[-1], dtype=cap_seq.dtype)  # only for lstm-dsa

            for i, index in enumerate(indices):
                feat_ids, cap_ids = index
                feat_bigids[total_ids: total_ids + len(feat_ids)] = total_query_ids + feat_ids
                cap_bigids[total_ids: total_ids + len(feat_ids)] = total_cap_ids + cap_ids
                new_hr_for_dsa[i, :len(feat_ids)] = proposals[i, feat_ids]
                new_seq_for_dsa[i, :len(feat_ids)] = cap_seq[total_cap_ids + cap_ids]
                total_query_ids += query_nums[i]
                total_cap_ids += gt_nums[i]
                total_ids += len(feat_ids)
        
        # Caption Prediction
        cap_probs = {}
        
        if captioner_type == 'none':
            # "bypass" when captioner model is not needed
            cost_caption = torch.zeros(N_, N_q, all_cap_num, device=proposals.device)  # [bs, queries, all_cap_num]
            loss_caption = torch.zeros(N_, N_q, all_cap_num, device=proposals.device) # [bs, queries, all_cap_num]
            cap_probs['cap_prob_train'] = torch.zeros(1, device=proposals.device)
            cap_probs['cap_prob_eval'] = torch.zeros(N_, N_q, 3, device=proposals.device)
            seq = torch.zeros(N_, N_q, 3, device=proposals.device)
            return cost_caption, loss_caption, cap_probs, seq
        
        elif captioner_type in ['light']:
            clip = hs_r.unsqueeze(1) # [total_valid_queries, 1, C]
            clip_mask = clip.new_ones(clip.shape[:2]) # [total_valid_queries, 1]
            event = None

            clip_ext = clip[feat_bigids]
            clip_mask_ext = clip_mask[feat_bigids]

            if self.training:
                seq = dt['cap_tensor'][cap_bigids]
                if self.opt.caption_cost_type != 'rl':
                    cap_prob = cap_head(event, clip_ext, clip_mask_ext, seq)
                    cap_probs['cap_prob_train'] = cap_prob
            else:
                with torch.no_grad():
                    seq_gt = dt['cap_tensor'][cap_bigids]
                    cap_prob = cap_head(event, clip_ext, clip_mask_ext, seq_gt)
                    seq, cap_prob_eval = cap_head.sample(event, clip, clip_mask)

                    if len(seq):
                        seq = seq.reshape(-1, N_q, seq.shape[-1])
                        cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
                    cap_probs['cap_prob_eval'] = cap_prob_eval

        # Loss and Cost Computation
        if self.opt.caption_cost_type == 'loss':
            cap_prob = cap_prob.reshape(-1, cap_prob.shape[-2], cap_prob.shape[-1]) # [match_pairs, max_caption_length, vocab_size]
            caption_tensor = dt['cap_tensor'][:, 1:][cap_bigids] # [match_pairs, max_caption_length]
            caption_mask = dt['cap_mask'][:, 1:][cap_bigids] # [match_pairs, max_caption_length]
            cap_loss = cap_head.build_loss(cap_prob, caption_tensor, caption_mask) # [match_pairs]
            cap_cost = cap_loss
        else:
            raise AssertionError('caption cost type error')

        if indices: # [(tensor([21, 31, 44, 54, 69, 76, 96]), tensor([6, 5, 4, 0, 3, 2, 1]))]
            return cap_loss.mean(), cap_probs, seq


        ## NOT USED?????????????????
        cap_id, query_id = cap_bigids, feat_bigids

        cost_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        cost_caption[query_id, cap_id] = cap_cost

        loss_caption = hs_r.new_zeros((max(query_id) + 1, max(cap_id) + 1))
        loss_caption[query_id, cap_id] = cap_loss
        cost_caption = cost_caption.reshape(-1, N_q, max(cap_id) + 1)  # batch_size * num_queries * all_caption_num
        loss_caption = loss_caption.reshape(-1, N_q, max(cap_id) + 1)

        return cost_caption, loss_caption, cap_probs, seq

    def caption_prediction_eval(self, cap_head, dt, queries, decoder_type, indices=None):
        # Ensures that no matching indices are provided in evaluation mode
        assert indices == None

        N_, N_q, C = queries.shape
        query_mask = torch.ones(N_, queries.shape[1], device=queries.device).bool()
        gt_mask = dt['gt_boxes_mask']
        mix_mask = torch.zeros(query_mask.sum().item(), gt_mask.sum().item())
        query_nums, gt_nums = query_mask.sum(1).cpu(), gt_mask.sum(1).cpu()
        hs_r = torch.masked_select(queries, query_mask.unsqueeze(-1)).reshape(-1, C) # [288, 512]

        # Evaluation Mode
        # Match all valid queries with all valid annotations in each sample of the batch
        row_idx, col_idx = 0, 0
        for i in range(N_):
            mix_mask[row_idx: (row_idx + query_nums[i]), col_idx: (col_idx + gt_nums[i])] = 1
            row_idx = row_idx + query_nums[i]
            col_idx = col_idx + gt_nums[i]

        cap_probs = {}

        if decoder_type in ['none']:
            cap_probs['cap_prob_train'] = torch.zeros(1, device=queries.device)
            cap_probs['cap_prob_eval'] = torch.zeros(N_, N_q, 3, device=queries.device)
            seq = torch.zeros(N_, N_q, 3, device=queries.device)
            return cap_probs, seq

        elif decoder_type in ['light']:
            clip = hs_r.unsqueeze(1) # [1, 288, 512]
            clip_mask = clip.new_ones(clip.shape[:2]) # [1, 288]
            event = None
            seq, cap_prob_eval = cap_head.sample(event, clip, clip_mask)
            if len(seq):
                seq = seq.reshape(-1, N_q, seq.shape[-1])
                cap_prob_eval = cap_prob_eval.reshape(-1, N_q, cap_prob_eval.shape[-1])
            cap_probs['cap_prob_eval'] = cap_prob_eval # [1, 288, 30]

        return cap_probs, seq


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    @torch.no_grad()
    def forward(self, outputs, target_sizes, loader):
        """ Perform the computation
        Sort based on pred_logits --> topk_values (scores), topk_indexes (query indices)
        seq --> Process to human-readable captions --> sort based on topk_boxes
        seq_score                                  --> sort based on topk_boxes
        --> return top-eseq_lens (model predict) caption, event info...

        Parameters:
            outputs: raw outputs of the model (all queries)
            ['pred_logits', 'pred_count', 'pred_boxes', 'caption_probs', 'seq', 'matched_indices']

            target_sizes: tensor of dimension [batch_size] containing the size of each video of the batch
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        N, N_q, N_class = out_logits.shape
        assert len(out_logits) == len(target_sizes)

        # Compute Scores and Indices
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), N_q, dim=1) # Selects the top num_queries
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2] # [batch_size, num_queries]
        labels = topk_indexes % out_logits.shape[2]
        # Process Bounding Boxes
        boxes = box_ops.box_cl_to_xy(out_bbox)
        raw_boxes = copy.deepcopy(boxes)
        boxes[boxes < 0] = 0 # Clamp Boxes to Valid Range
        boxes[boxes > 1] = 1
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2)) # Selects the boxes corresponding to the top-k query indices (topk_boxes) to align with the highest-confidence predictions
        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        seq = outputs['seq']  # caption sequences, [batch_size, num_queries, max_Cap_len=30]
        cap_prob = outputs['caption_probs']['cap_prob_eval']  # their probabilities, [batch_size, num_queries, max_Cap_len=30]
        eseq_lens = outputs['pred_count'].argmax(dim=-1).clamp(min=1)

        if len(seq):
            mask = (seq > 0).float() # binary mask where valid tokens (non-zero) are 1, and padding tokens are 0
            cap_scores = (mask * cap_prob).sum(2).cpu().numpy().astype('float')
            seq = seq.detach().cpu().numpy().astype('int')  # (batch_size, eseq_len, cap_len)
            caps = [[loader.dataset.translator.rtranslate(s) for s in s_vid] for s_vid in seq]
            caps = [[caps[batch][idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
            cap_scores = [[cap_scores[batch, idx] for q_id, idx in enumerate(b)] for batch, b in enumerate(topk_boxes)]
            # reorder captions and cap_scores so they match the order of the queries selected in topk_boxes
        else:
            bs, num_queries = boxes.shape[:2]
            cap_scores = [[-1e5] * num_queries] * bs
            caps = [[''] * num_queries] * bs

        # Includes all queries predictions, with eseq_lens (model prediction) for downstream filtering
        results = [
            {'scores': s, 'labels': l, 'boxes': b, 'raw_boxes': b, 'captions': c, 'caption_scores': cs, 'query_id': qid,
             'vid_duration': ts, 'pred_seq_len': sl} for s, l, b, rb, c, cs, qid, ts, sl in
            zip(scores, labels, boxes, raw_boxes, caps, cap_scores, topk_boxes, target_sizes, eseq_lens)]
        return results


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    device = torch.device(args.device)
    base_encoder = build_base_encoder(args)
    decoder = build_decoder(args)
    processor = build_processor(args)
    captioner = build_captioner(args)

    model = PDVC(
        base_encoder,
        decoder,
        processor,
        captioner,
        num_classes=args.num_classes,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        opt=args
    )

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, # 2
                   'loss_bbox': args.bbox_loss_coef, # 0
                   'loss_giou': args.giou_loss_coef, # 4
                   'loss_counter': args.count_loss_coef, # 0.5
                   'loss_caption': args.caption_loss_coef, # 2
                   }

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']

    criterion = SetCriterion(args.num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha,
                             focal_gamma=args.focal_gamma, opt=args)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess(args)}

    return model, criterion, postprocessors
