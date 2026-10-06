import torch
import torch.nn.functional as F

AXES = (0, 2, 3)
EPSILON = 1e-12

def scnp(logits, y_true, weight, neighborhood_size=3):
    """
    Same Class Neighbor Penalization: replaces each logit by the worst logit among its
    same-class neighbours. `weight` limits this to fully annotated neighbourhoods.
    See https://jmlipman.github.io/SCNP-SameClassNeighborPenalization/
    """

    kernel = (neighborhood_size, neighborhood_size)
    stride = (1, 1)
    padding = (neighborhood_size // 2, neighborhood_size // 2)

    foreground = -F.max_pool2d(-(logits * y_true + 9999 * (1 - y_true)), kernel, stride, padding)
    background = F.max_pool2d(logits * (1 - y_true) - 9999 * y_true, kernel, stride, padding)

    penalized = foreground * y_true + background * (1 - y_true)

    annotated = -F.max_pool2d(-weight, kernel, stride, padding)
    return penalized * annotated + logits * (1 - annotated)

def _weighted_mean(values, weight):
    return torch.sum(weight * values, axis=AXES) / torch.sum(weight, axis=AXES)

def _confusion(y_pred, y_true, weight):
    tp = _weighted_mean(y_true * y_pred, weight)
    fp = _weighted_mean((1 - y_true) * y_pred, weight)
    fn = _weighted_mean((1 - y_pred) * y_true, weight)
    tn = _weighted_mean((1 - y_pred) * (1 - y_true), weight)
    return tp, fp, fn, tn

def crossentropy_loss(y_pred, y_true, weight):
    return torch.mean(-_weighted_mean(y_true * torch.log(y_pred + EPSILON), weight))

def dice_loss(y_pred, y_true, weight):
    tp, fp, fn, _ = _confusion(y_pred, y_true, weight)
    return 1 - torch.mean((2 * tp + EPSILON) / (2 * tp + fp + fn + EPSILON))

def mcc_loss(y_pred, y_true, weight):
    tp, fp, fn, tn = _confusion(y_pred, y_true, weight)
    num = (tp * tn) - (fp * fn)
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**0.5
    return 1 - torch.mean((num + EPSILON) / (den + EPSILON))

def dice_ce_loss(y_pred, y_true, weight):
    return dice_loss(y_pred, y_true, weight) + crossentropy_loss(y_pred, y_true, weight)

def mcc_ce_loss(y_pred, y_true, weight):
    return mcc_loss(y_pred, y_true, weight) + crossentropy_loss(y_pred, y_true, weight)

LOSS_OPTIONS = {
    crossentropy_loss: 'CE',
    dice_loss: 'Dice',
    mcc_loss: 'MCC',
    dice_ce_loss: 'Dice+CE',
    mcc_ce_loss: 'MCC+CE',
}
