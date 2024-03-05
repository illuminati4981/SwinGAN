import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from lpips import LPIPS


class MultiscaleLPIPS:
    def __init__(
        self,
        min_loss_res: int = 16,
        level_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        l1_weight: float = 0.1
    ):
        super().__init__()
        self.min_loss_res = min_loss_res
        self.weights = level_weights
        self.l1_weight = l1_weight
        self.lpips_network = LPIPS(net="vgg", verbose=False).cuda()

    def measure_lpips(self, x, y, mask):
        if mask is not None:
            # To avoid biasing the results towards black pixels, but random noise in the masked areas
            noise = (torch.randn_like(x) + 0.5) / 2.0
            x = x + noise * (1.0 - mask)
            y = y + noise * (1.0 - mask)

        return self.lpips_network(x, y, normalize=True).mean() 

    def __call__(self, f_hat, x_clean: Tensor, y: Tensor, mask = None):
        x = f_hat(x_clean)

        losses = []

        if mask is not None:
            mask = F.interpolate(mask, y.shape[-1], mode="area")

        for weight in self.weights:
            # At extremely low resolutions, LPIPS stops making sense, so omit those
            if y.shape[-1] <= self.min_loss_res:
                break
            
            if weight > 0:
                loss = self.measure_lpips(x, y, mask)
                losses.append(weight * loss)

            if mask is not None:
                mask = F.avg_pool2d(mask, 2)

            x = F.avg_pool2d(x, 2)
            x_clean = F.avg_pool2d(x_clean, 2)
            y = F.avg_pool2d(y, 2)
        
        total = torch.stack(losses).sum(dim=0) if len(losses) > 0 else 0.0
        l1 = self.l1_weight * F.l1_loss(x, y)

        return total + l1