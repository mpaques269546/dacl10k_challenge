import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
import torch.distributed as dist
from torch.cuda.amp import autocast
import torchvision




def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    # fix type mismatch
    point_coords = point_coords.type_as(input)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    
    hw = inputs.shape[1] if inputs.shape[-1]>0 else 1
    

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction='none')

    loss = torch.einsum('nc,mc->nm', pos, targets) + torch.einsum(
        'nc,mc->nm', neg, (1 - targets))
    return loss / hw


class MaskHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network for segmentation

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_mask: float = 1,
                 cost_dice: float = 1,
                 num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, 'all costs cant be 0'

        self.num_points = num_points
    

    @torch.no_grad()
    def forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs['pred_logits'].shape[:2]

        indices = []
        # Iterate through batch size
        for b in range(bs):

            out_prob = outputs['pred_logits'][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]['labels']
            
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs['pred_masks'][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]['masks'].to(out_mask)
            
            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(
                1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)
            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                # jit not match some torch versions
                # cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)

                # Compute the dice loss betwen masks
                # cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
                cost_dice = batch_dice_loss(out_mask, tgt_mask)

            # Final cost matrix
            C = (
                self.cost_mask * cost_mask + self.cost_class * cost_class +
                self.cost_dice * cost_dice)
            C = C.reshape(num_queries, -1).cpu()
            indices.append(linear_sum_assignment(C))

        return [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]





def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):

    def __init__(self, tensors, mask=None ):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[:img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)






class Mask2FormerLoss(nn.Module):
    """This class computes the loss for Mask2former.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes:int=19,  weight_dict:dict={'loss_mask':1., 'loss_dice':1., 'loss_ce':0.4} ,  losses:list=['labels', 'masks' ], aux_loss_weight:float=0.1 ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = MaskHungarianMatcher() #  matcher: module able to compute a matching between targets and proposals
        self.weight_dict = weight_dict
        self.aux_loss_weight = aux_loss_weight
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device)
        target_classes[idx] = target_classes_o.to(src_logits.device)
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce }
        return losses
    
    def dice_loss_(self, inputs: torch.Tensor,targets: torch.Tensor):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        
        masks = [t['masks'] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = F.interpolate(src_masks.unsqueeze(0), scale_factor=4 , mode='bilinear').squeeze(0)
        
        sigmoid_ce_loss = F.binary_cross_entropy_with_logits(src_masks , target_masks, reduction='none')
        sigmoid_ce_loss = sigmoid_ce_loss.mean() # sigmoid_ce_loss.mean(1).sum() / num_masks
        
        dice_loss = self.dice_loss_(src_masks , target_masks)
        dice_loss = dice_loss.mean() #dice_loss.sum() / num_masks
        
        losses = {'loss_mask': sigmoid_ce_loss  ,'loss_dice': dice_loss }
        del src_masks , target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks)
    

    def preprocess_gt_semantic(self, targets ):
        [B,C,H,W] = targets.shape
        new_targets = []
        for b in range(B):
            mask = targets[b]
            masks , labels  = [],[]
            for c in range(C):
                if mask[c].sum()>0:
                    masks.append(mask[c])
                    labels.append(torch.tensor([c]))
            if len(masks)==0:
                masks = [torch.zeros_like(mask[c])]
                labels = [torch.tensor([C])]
            masks = torch.stack(masks, dim=0)
            labels = torch.stack(labels, dim=0).squeeze(-1)
            new_targets.append({'labels': labels,'masks': masks})
        return new_targets

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # out['pred_logits']: [bs, n_mask, n_cls+1] , out['pred_masks']:[bs, n_mask, h//4, w//4])
        targets = self.preprocess_gt_semantic(targets)
        
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = torch.as_tensor([num_masks],
                                    dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if dist.is_available() and dist.is_initialized():
            torch.distributed.all_reduce(num_masks)
        #rank, world_size = get_dist_info()
        num_masks = torch.clamp(num_masks  , min=1).item() # torch.clamp(num_masks / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        num_aux = 0
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                num_aux +=1
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_masks)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        #print(losses)
        loss = losses['loss_ce'] * self.weight_dict['loss_ce'] + losses['loss_mask'] * self.weight_dict['loss_mask'] + losses['loss_dice'] * self.weight_dict['loss_dice']
        aux_loss = [losses['loss_ce_'+str(i)] * self.weight_dict['loss_ce'] + losses['loss_mask_'+str(i)] * self.weight_dict['loss_mask'] + losses['loss_dice_'+str(i)] * self.weight_dict['loss_dice'] for i in range(num_aux)]
        final_loss = loss + self.aux_loss_weight * sum(aux_loss) / num_aux
        return final_loss
























class Mask2FormerLoss_old(nn.Module):
    """This class computes the loss for Mask2former.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes:int=19,  weight_dict:dict={'loss_mask':1., 'loss_dice':1., 'loss_ce':1.} ,  losses:list=['labels', 'masks' ],):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = MaskHungarianMatcher() #  matcher: module able to compute a matching between targets and proposals
        self.weight_dict = weight_dict
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)]).to(device=src_logits.device)
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses
    
    def dice_loss_(self, inputs: torch.Tensor,targets: torch.Tensor):
        """
        Compute the DICE loss, similar to generalized IOU for masks
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                     classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
        """
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert 'pred_masks' in outputs
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_masks']
        src_masks = src_masks[src_idx]
        
        masks = [t['masks'] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        src_masks = F.interpolate(src_masks.unsqueeze(0), scale_factor=4 , mode='bilinear').squeeze(0)
        
        sigmoid_ce_loss = F.binary_cross_entropy_with_logits(src_masks , target_masks, reduction='none')
        sigmoid_ce_loss = sigmoid_ce_loss.mean(1).sum() / num_masks
        
        dice_loss = self.dice_loss_(src_masks , target_masks)
        dice_loss = dice_loss.sum() / num_masks
        
        losses = {'loss_mask': sigmoid_ce_loss,'loss_dice': dice_loss}
        del src_masks , target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_masks)
    

    def preprocess_gt_semantic(self, targets ):
        [B,C,H,W] = targets.shape
        new_targets = []
        for b in range(B):
            mask = targets[b]
            masks , labels  = [],[]
            for c in range(C):
                if mask[c].sum()>0:
                    masks.append(mask[c])
                    labels.append(torch.tensor([c]))
            if len(masks)==0:
                #print('WARNING null mask')
                masks.append( torch.zeros_like(mask[c]))
                labels.append( torch.tensor([19]))
            assert len(masks)>0
            masks = torch.stack(masks, dim=0)
            labels = torch.stack(labels, dim=0).squeeze(-1)
            new_targets.append({'labels': labels,'masks': masks})
        return new_targets

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # out['pred_logits']: [bs, n_mask, n_cls+1] , out['pred_masks']:[bs, n_mask, h//4, w//4])
        targets = self.preprocess_gt_semantic(targets)
        
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t['labels']) for t in targets)
        num_masks = torch.as_tensor([num_masks],
                                    dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if dist.is_available() and dist.is_initialized():
            torch.distributed.all_reduce(num_masks)
        #rank, world_size = get_dist_info()
        num_masks = torch.clamp(num_masks  , min=1).item() # torch.clamp(num_masks / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_masks))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_masks)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        loss = sum(list(losses.values()))/len(losses)
        return loss
