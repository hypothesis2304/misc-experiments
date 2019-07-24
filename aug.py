import numpy as np
import torch
from torch.distributions import uniform
from utils import get_predictions, generate_bbox

def cutmix(model, s_x, s_y, t_x, alpha, num_class):
    x_aug = s_x.clone()
    temp_tx = t_x.clone()
    t_y = get_predictions(model, t_x, num_class, temperature)
    mu = np.random.beta(alpha, alpha)
    lmbd = np.maximum(mu, 1-mu)
    lmbd = torch.tensor(lmbd)
    bbx1, bby1, bbx2, bby2 = generate_bbox(s_x.size(), lmbd)
    x_aug[:, :, bbx1:bbx2, bby1:bby2] = temp_tx[:, :, bbx1:bbx2, bby1:bby2]
    y_aug = (lmbd * s_y.float()) + ((1 - lmbd) * t_y.float())
    return x_aug.cuda(), y_aug.cuda()

# def mixup(model, s_x, s_y, t_x, alpha, num_class):
#     t_y = get_predictions(model, t_x, num_class)
#     mu = np.random.beta(alpha, alpha, size=(s_x.size(0)))
#     lmbd = np.maximum(mu, 1 - mu)
#     lmbdx = torch.from_numpy(lmbd.reshape(lmbd.shape[0],1,1,1)).float().to('cuda')
#     lmbdy = torch.from_numpy(lmbd.reshape(lmbd.shape[0],1)).float().to('cuda')
#     s_y = s_y.float()
#     t_y = t_y.float()
#     x_mix = (s_x * lmbdx) + (t_x * (1 - lmbdx))
#     y_mix = (s_y * lmbdy) + (t_y * (1 - lmbdy))
#     return x_mix.cuda(), y_mix.cuda()

def mixup(model, s_x, s_y, t_x, alpha, num_class, temperature):
    t_y = get_predictions(model, t_x, num_class, temperature)
    beta_distribution = torch.distributions.Beta(alpha, alpha)
    lambdas_mu = beta_distribution.sample([s_x.shape[0]]).to('cuda')
    lambdas_mu = torch.max(lambdas_mu, 1-lambdas_mu)
    lambdas_mu_features = lambdas_mu.view(lambdas_mu.shape[0], 1, 1, 1)
    lambdas_mu_labels = lambdas_mu.view(lambdas_mu.shape[0], 1)
    x_aug = s_x * lambdas_mu_features + t_x * (1 - lambdas_mu_features)
    y_aug = s_y * lambdas_mu_labels + t_y * (1 - lambdas_mu_labels)
    return x_aug, y_aug
