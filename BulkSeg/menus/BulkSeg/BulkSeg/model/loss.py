import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def cellpose_loss(output, target):
    loss1 = F.mse_loss(output[:, :2, :, :], target[:, :2, :, :]) / 2.0
    loss2 = F.binary_cross_entropy_with_logits(output[:, 2, :, :],
        (target[:, 2, :, :]>0.5).float())
    loss = loss1 + loss2
    return loss