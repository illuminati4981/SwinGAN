from typing import Any, overload, Dict, Union, List, Sequence
import math
import random

import torch
from torch.nn import functional as F
from torchvision.transforms import RandomCrop, CenterCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, Compose
import numpy as np

# FIXME call from other dir need add degradation.
from degradation.utils.image import USMSharp, DiffJPEG, filter2D
from degradation.utils.degradation import (
    random_add_gaussian_noise_pt, random_add_poisson_noise_pt, circular_lowpass_kernel, random_mixed_kernels
)


class BatchTransform:
    
    @overload
    def __call__(self, batch: Any) -> Any:
        ...


class IdentityBatchTransform(BatchTransform):
    
    def __call__(self, batch: Any) -> Any:
        return batch


class RealESRGANBatchTransform(BatchTransform):
    """
    It's too slow to process a batch of images under RealESRGAN degradation
    model on CPU (by dataloader), which may cost 0.2 ~ 1 second per image.
    So we execute the degradation process on GPU after loading a batch of images
    and kernels from dataloader.
    """
    def __init__(
        self,
        # Params for cropping and kernels
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        use_rot: bool,
        ## blur kernel settings of the first degradation stage
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        betag_range: Sequence[float],
        betap_range: Sequence[float],
        sinc_prob: float,
        ## blur kernel settings of the second degradation stage
        blur_kernel_size2: int,
        kernel_list2: Sequence[str],
        kernel_prob2: Sequence[float],
        blur_sigma2: Sequence[float],
        betag_range2: Sequence[float],
        betap_range2: Sequence[float],
        sinc_prob2: float,
        final_sinc_prob: float,
        # Params for transform, the high order degradation process
        use_sharpener: bool,
        resize_hq: bool,
        queue_size: int,
        resize_prob: Sequence[float],
        resize_range: Sequence[float],
        gray_noise_prob: float,
        gaussian_noise_prob: float,
        noise_range: Sequence[float],
        poisson_scale_range: Sequence[float],
        jpeg_range: Sequence[int],
        second_blur_prob: float,
        stage2_scale: Union[float, Sequence[Union[float, int]]],
        resize_prob2: Sequence[float],
        resize_range2: Sequence[float],
        gray_noise_prob2: float,
        gaussian_noise_prob2: float,
        noise_range2: Sequence[float],
        poisson_scale_range2: Sequence[float],
        jpeg_range2: Sequence[int]
    ) -> "RealESRGANBatchTransform":
        super().__init__()

        # -------- cropping, kernels params -------- #
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["center", "random", "none"], f"invalid crop type: {self.crop_type}"

        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        # a list for each kernel probability
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        # betag used in generalized Gaussian blur kernels
        self.betag_range = betag_range
        # betap used in plateau blur kernels
        self.betap_range = betap_range
        # the probability for sinc filters
        self.sinc_prob = sinc_prob

        self.blur_kernel_size2 = blur_kernel_size2
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2
        
        # a final sinc filter
        self.final_sinc_prob = final_sinc_prob
        
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        
        # kernel size ranges from 11 to 29
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # TODO: kernel range is now hard-coded, should be in the configure file, Just let here be hard-coded
        # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

        # -------- Degradation process params -------- #

        # resize settings for the first degradation process
        self.resize_prob = resize_prob
        self.resize_range = resize_range
        
        # noise settings for the first degradation process
        self.gray_noise_prob = gray_noise_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_range = noise_range
        self.poisson_scale_range = poisson_scale_range
        self.jpeg_range = jpeg_range
        
        self.second_blur_prob = second_blur_prob
        self.stage2_scale = stage2_scale
        assert (
            isinstance(stage2_scale, (float, int)) or (
                isinstance(stage2_scale, Sequence) and len(stage2_scale) == 2 and 
                all(isinstance(x, (float, int)) for x in stage2_scale)
            )
        ), f"stage2_scale can not be {type(stage2_scale)}"
        
        # resize settings for the second degradation process
        self.resize_prob2 = resize_prob2
        self.resize_range2 = resize_range2
        
        # noise settings for the second degradation process
        self.gray_noise_prob2 = gray_noise_prob2
        self.gaussian_noise_prob2 = gaussian_noise_prob2
        self.noise_range2 = noise_range2
        self.poisson_scale_range2 = poisson_scale_range2
        self.jpeg_range2 = jpeg_range2
        
        self.use_sharpener = use_sharpener
        if self.use_sharpener:
            self.usm_sharpener = USMSharp()
        else:
            self.usm_sharpener = None
        self.resize_hq = resize_hq
        self.queue_size = queue_size
        self.jpeger = DiffJPEG(differentiable=False)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, "queue_lr"):
            # TODO: Being multiple of batch_size seems not necessary for queue_size
            assert self.queue_size % b == 0, f"queue size {self.queue_size} should be divisible by batch size {b}"
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).to(self.lq)
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).to(self.lq)
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    # Customized to parse the batch of hq img to such a format
    def _parse(self, batch_hq_img: torch.Tensor) ->  Dict[str, Union[torch.Tensor, str]]:
        # crop, and generate kernel for each image 
        img_hq = None
        _kernel = []
        _kernel2 = []
        _sinc_kernel = []

        # list for the crop + arugmentation
        transform_list = []

        # Crop
        if self.crop_type == "random":
            transform_list.append(RandomCrop(self.out_size))
        elif self.crop_type == "center":
            transform_list.append(CenterCrop(self.out_size))

        # arugmentation
        if self.use_hflip:
            transform_list.append(RandomHorizontalFlip())
        if self.use_rot:
            transform_list.append(RandomVerticalFlip())
            transform_list.append(RandomRotation())

        # Define the transform
        transform = Compose(transform_list)
        img_hq = torch.stack([transform(img) for img in batch_hq_img]) 

        batch_size = batch_hq_img.size()[0]
        for _ in range(batch_size):
            kernel, kernel2, sinc_kernel = None, None, None

            # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
            kernel_size = self.blur_kernel_size
            if np.random.uniform() < self.sinc_prob:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None
                )
            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

            # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
            kernel_size = self.blur_kernel_size2
            if np.random.uniform() < self.sinc_prob2:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2,
                    self.kernel_prob2,
                    kernel_size,
                    self.blur_sigma2,
                    self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None
                )

            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

            # ------------------------------------- the final sinc kernel ------------------------------------- #
            if np.random.uniform() < self.final_sinc_prob:
                # kernel_size = random.choice(self.kernel_range)
                kernel_size = 21 # FIXME : hard-coded to 21, since ensure the kerels size is consistent within a batch
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor

            kernel = torch.FloatTensor(kernel)
            kernel2 = torch.FloatTensor(kernel2)

            _kernel.append(kernel)
            _kernel2.append(kernel2)
            _sinc_kernel.append(sinc_kernel)

        return {
            "hq": img_hq, "kernel1": torch.stack(_kernel), "kernel2": torch.stack(_kernel2),
            "sinc_kernel": torch.stack(_sinc_kernel), "txt": ""
        }

    @torch.no_grad()
    def __call__(self, batch_hq_img: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[str]]]:

        # print(type(batch_hq_img))
        batch = self._parse(batch_hq_img)
        # training data synthesis
        hq = batch["hq"]
        if self.use_sharpener:
            self.usm_sharpener.to(hq)
            hq = self.usm_sharpener(hq)
        self.jpeger.to(hq)
        
        kernel1 = batch["kernel1"]
        kernel2 = batch["kernel2"]
        sinc_kernel = batch["sinc_kernel"]

        ori_h, ori_w = hq.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(hq, kernel1)
        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.resize_prob)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range, clip=True,
                rounds=False, gray_prob=self.gray_noise_prob
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range,
                gray_prob=self.gray_noise_prob,
                clip=True,
                rounds=False
            )
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
        # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = torch.clamp(out, 0, 1)
        out = self.jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
            out = filter2D(out, kernel2)
        
        # select scale of second degradation stage
        if isinstance(self.stage2_scale, Sequence):
            min_scale, max_scale = self.stage2_scale
            stage2_scale = np.random.uniform(min_scale, max_scale)
        else:
            stage2_scale = self.stage2_scale
        stage2_h, stage2_w = int(ori_h / stage2_scale), int(ori_w / stage2_scale)
        # print(f"stage2 scale = {stage2_scale}")
        
        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.resize_prob2)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out, size=(int(stage2_h * scale), int(stage2_w * scale)), mode=mode
        )
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=self.noise_range2, clip=True,
                rounds=False, gray_prob=self.gray_noise_prob2
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=self.gray_noise_prob2,
                clip=True,
                rounds=False
            )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = filter2D(out, sinc_kernel)

        # resize back to gt_size since We are doing restoration task
        if stage2_scale != 1:
            out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")
        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255).float()
        hq = torch.clamp((hq * 255.0).round(), 0, 255).float()
        
        if self.resize_hq and stage2_scale != 1:
            # resize hq
            hq = F.interpolate(hq, size=(stage2_h, stage2_w), mode="bicubic", antialias=True)
            hq = F.interpolate(hq, size=(ori_h, ori_w), mode="bicubic", antialias=True)
        
        self.gt = hq
        self.lq = lq
        self._dequeue_and_enqueue()

        # [0, 1], float32, rgb, nhwc
        lq = self.lq.float().permute(0, 2, 3, 1).contiguous()
        # [-1, 1], float32, rgb, nhwc
        hq = (self.gt).float().permute(0, 2, 3, 1).contiguous()
        
        return dict(jpg=hq, hint=lq, txt=batch["txt"])
