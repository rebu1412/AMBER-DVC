# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn

from misc.detr_utils import box_ops
from misc.detr_utils.misc import (accuracy, get_world_size,
                         is_dist_avail_and_initialized)

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, focal_gamma=2, opt={}):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.opt = opt
        # the frequency of the event number of the dataset
        # [0.099458, 0.099458, 0.120662, 0.150155, 0.120104, 0.107265, 0.104614, 0.102564, 0.101135, 0.100245, 0.100105, 
        # 0.099947, 0.099768, 0.099736, 0.099595, 0.099587, 0.099587, 0.099578, 0.099582, 0.099586, 0.099582, 0.099458, 
        # 0.099578, 0.099458, 0.099458, 0.099458, 0.099573, 0.099573]
        counter_class_rate = [0.00000000e+00, 0.00000000e+00, 1.93425917e-01, 4.12129084e-01,
       1.88929963e-01, 7.81296833e-02, 5.09541413e-02, 3.12718553e-02,
       1.84833650e-02, 8.39244680e-03, 6.59406534e-03, 4.49595364e-03,
       2.19802178e-03, 1.79838146e-03, 5.99460486e-04, 4.99550405e-04,
       4.99550405e-04, 1.99820162e-04, 2.99730243e-04, 3.99640324e-04,
       2.99730243e-04, 0.00000000e+00, 1.99820162e-04, 0.00000000e+00,
       0.00000000e+00, 0.00000000e+00, 9.99100809e-05, 9.99100809e-05]
        self.counter_class_rate = torch.tensor(counter_class_rate)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        indices, many2one_indices = indices # [(tensor_i=[0, 2, 5], tensor_j=[1, 0, 2]), ...]

        # Classification loss
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'] # [bs, num_queries, 1]
        idx = self._get_src_permutation_idx(indices) # (tensor([0, 0, 0, 0, 0, 0, 0]), tensor([54, 56, 74, 76, 83, 87, 96]))
        # Generate target label tensor
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device) # [bs, queries], initialize all values ​​as self.num_classes ("no-object" class).
        target_classes[idx] = target_classes_o # [bs, queries], match queries --> 0 foreground; else --> 1 background
        # Convert labels to one-hot encoding
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1], # [bs, num_queries, num_classes + 1]
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:,:,:-1] # [bs, num_queries, 1]
        
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        # Counter LOSS
        pred_count = outputs['pred_count'] # [bs, opt.max_eseq_length]
        max_length = pred_count.shape[1] - 1
        # Create a Counter target
        counter_target = [len(target['boxes']) if len(target['boxes']) < max_length  else max_length for target in targets]
        counter_target = torch.tensor(counter_target, device=src_logits.device, dtype=torch.long)
        counter_target_onehot = torch.zeros_like(pred_count) # [bs, opt.max_eseq_length]
        counter_target_onehot.scatter_(1, counter_target.unsqueeze(-1), 1) # One_hot: [bs, opt.max_eseq_length]
        weight = self.counter_class_rate[:max_length + 1].to(src_logits.device)
        counter_loss = cross_entropy_with_gaussian_mask(pred_count, counter_target_onehot, self.opt, weight)
        losses['loss_counter'] = counter_loss

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, 
        ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        
        pred_logits = outputs['pred_logits'] # [bs, num_queries, 1]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 2]
           The target boxes are expected in format (center, length), normalized by the image size.
        """
        
        indices, many2one_indices = indices # [(tensor_i=[0, 2, 5], tensor_j=[1, 0, 2]), ...]
        N = len(indices[-1][0])
        assert 'pred_boxes' in outputs
        losses = {}

        idx, idx2 = self._get_src_permutation_idx2(indices) # (tensor([bs, bs, bs, ...]), tensor_i([0, 2, 5]))
        src_boxes = outputs['pred_boxes'][idx] # [num_matched_boxes, 2]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # [num_matched_boxes, 2]

        # Print a random matched pair in normalized [center, length] format
        if len(src_boxes) > 0:  # Ensure there's at least one matched pair
            random_idx = torch.randint(0, len(src_boxes), (1,)).item()  # Select a random index
            pred_box = src_boxes[random_idx]  # [center, length]
            tgt_box = target_boxes[random_idx]  # [center, length]
            print(f" [{pred_box[0]:.4f}, {pred_box[1]:.4f}] | [{tgt_box[0]:.4f}, {tgt_box[1]:.4f}]")

        # L1 regression loss (Make sure the predicted box coordinates are close to ground-truth (L1 loss).)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # Generalized IoU loss (Make sure the prediction box is similar in shape and position to the ground-truth)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cl_to_xy(src_boxes),
            box_ops.box_cl_to_xy(target_boxes))) # The higher the GIoU (better box match), the lower the loss
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # Self-IoU loss (Minimize overlap between prediction boxes in the same video)
        self_iou = torch.triu(box_ops.box_iou(box_ops.box_cl_to_xy(src_boxes),
                                              box_ops.box_cl_to_xy(src_boxes))[0], diagonal=1) # [num_matched_boxes, num_matched_boxes]
        sizes = [len(v[0]) for v in indices]
        self_iou_split = 0
        for i, c in enumerate(self_iou.split(sizes, -1)): # Split by Col
            cc = c.split(sizes, -2)[i] # Split by row
            self_iou_split += cc.sum() / (0.5 * (sizes[i]) * (sizes[i]-1)) # Number of pairs of non-repeating boxes= n(n+1)/2
        losses['loss_self_iou'] = self_iou_split

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_src_permutation_idx2(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        src_idx2 = torch.cat([src for (_, src) in indices])
        return (batch_idx, src_idx), src_idx2

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
                    ['pred_logits', 'pred_count', 'pred_boxes', 'caption_probs', 'seq', 'aux_outputs']

             targets: list of dicts, such that len(targets) == batch_size.
                    [
                        {
                        'boxes': [gt_num, 2]
                        'labels': [gt_num], 
                        'masks': None, 
                        'image_id': '...'
                        },
                    ]
                    The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        last_indices = self.matcher(outputs_without_aux, targets) # ([1_to_1,...], [1_to_many,...])
        outputs['matched_indices'] = last_indices

        num_boxes = sum(len(t["labels"]) for t in targets) # total gt boxes
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses: # ["labels", "boxes", "cardinality"]
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, last_indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            aux_indices = []
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                aux_indices.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    
            return losses, last_indices, aux_indices
                    # losses: 'loss_ce', 'loss_counter', 'loss_bbox', 'loss_giou', 'loss_self_iou', 'cardinality_error'
                    #         'loss_ce_0', 'loss_counter_0', 'loss_bbox_0', 'loss_giou_0', 'loss_self_iou_0', 'cardinality_error_0'
        
        return losses, last_indices
                # losses: 'loss_ce', 'loss_counter', 'loss_bbox', 'loss_giou', 'loss_self_iou', 'cardinality_error'
                # last_indices: ([1_to_1,...], [1_to_many,...])

def cross_entropy_with_gaussian_mask(inputs, targets, opt, weight):
    gau_mask = opt.lloss_gau_mask # Boolean, enable/disable Gaussian mask
    beta = opt.lloss_beta # adjust the strength of the Gaussian mask.

    N_, max_seq_len = targets.shape

    # [max_seq_len, max_seq_len]
    # [[0, 1, 2, 3, 4, 5, ...], [0, 1, 2, 3, 4, 5, ...], ...]
    gassian_mu = torch.arange(max_seq_len, device=inputs.device).unsqueeze(0).expand(max_seq_len, max_seq_len).float()
    
    # [[0, 0, 0, ...], [1, 1, 1, ...], ...]
    x = gassian_mu.transpose(0, 1)
    gassian_sigma = 2
    mask_dict = torch.exp(-(x - gassian_mu) ** 2 / (2 * gassian_sigma ** 2))
    _, ind = targets.max(dim=1)
    mask = mask_dict[ind]

    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight= 1 - weight)
    if gau_mask:
        coef = targets + ((1 - mask) ** beta) * (1 - targets)
    else:
        coef = targets + (1 - targets)
    loss = loss * coef
    loss = loss.mean(1)
    return loss.mean()

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def regression_loss(inputs, targets, opt, weight):
    inputs = F.relu(inputs) + 2
    max_id = torch.argmax(targets, dim=1)
    if opt.regression_loss_type == 'l1':
        loss = nn.L1Loss()(inputs[:, 0], max_id.float())
    elif opt.regression_loss_type == 'l2':
        loss = nn.MSELoss()(inputs[:, 0], max_id.float())
    return loss
