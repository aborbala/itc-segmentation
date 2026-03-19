# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import (
    RotatedTaskAlignedAssigner,
    TaskAlignedAssigner,
    dist2bbox,
    dist2rbox,
    make_anchors,
)
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.

    Args:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma=2.0, alpha=0.75):
        """Initialize the VarifocalLoss class."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score, gt_score, label):
        """Compute varifocal loss between predictions and ground truth."""
        # print("➡️ DEBUG Entering VarifocalLoss")

        weight = (
            self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label)
            + gt_score * label
        )
        with autocast(enabled=False):
            loss = (
                (
                    F.binary_cross_entropy_with_logits(
                        pred_score.float(), gt_score.float(), reduction="none"
                    )
                    * weight
                )
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Args:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float | list): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma=1.5, alpha=0.25):
        """Initialize FocalLoss class with no parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred, label):
        """Calculate focal loss with modulating factors for class imbalance."""
        # print("➡️ DEBUG Entering FocalLoss")

        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        # print("➡️ DEBUG Entering DFLoss")

        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape)
            * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape)
            * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        """Compute IoU and DFL losses for bounding boxes."""
        # print("➡️ DEBUG  Entering BboxLoss")
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points, target_bboxes, self.dfl_loss.reg_max - 1
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        """Compute IoU and DFL losses for rotated bounding boxes."""
        # print("➡️ DEBUG  Entering RotatedBboxLoss")

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points,
                xywh2xyxy(target_bboxes[..., :4]),
                self.dfl_loss.reg_max - 1,
            )
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        # print("➡️ DEBUG Entering KeypointLoss")

        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (
            pred_kpts[..., 1] - gt_kpts[..., 1]
        ).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        print("➡️  Entering v8DetectionLoss")
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]),
            1,
        )
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
        # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

        # ADDED: define ignore class
        self.ignore_class = 0

    def __call__(self, preds, batch):
        """Calculate and return the combined loss for detection and segmentation."""
        # --- ADDED: Ignore map for pixel-wise ignore during loss ---
        # print("➡️ DEBUG Entering v8SegmentationLoss; batch keys:", list(batch.keys()))
        ignore_map = batch["ignore_map"].to(self.device)  # [B, H, W], values 0 or 1

        # print(
        #     f"DEBUG [Loss __call__] ignore_map shape={ignore_map.shape}, "
        #     f"sum={ignore_map.sum().item()}/{ignore_map.numel()}"
        # )

        # --- ADDED: Remove class-0 GT objects before assignment ---
        # batch_idx = batch["batch_idx"].view(-1, 1)  # [N]
        gt_batch_idx = batch["batch_idx"].view(-1, 1)  # [sum_labels,1], save GT-level
        cls = batch["cls"].view(-1, 1)  # [N]
        bboxes = batch["bboxes"].view(-1, 4)  # [N, 4]
        targets = torch.cat((gt_batch_idx, cls, bboxes), dim=1)  # [N,6]
        masks = batch["masks"]  # [N, H, W] or [(sum_labels),H,W]

        # # A: Filter out class 0 targets before assignment:
        keep_mask = targets[:, 1] != self.ignore_class
        num_before = targets.shape[0]
        num_masks_before = masks.shape[0]
        num_class0 = (targets[:, 1] == self.ignore_class).sum().item()

        # print(
        #     f"DEBUG before filtering: {num_before} targets ({num_class0} class-0), "
        #     f"{num_masks_before} masks"
        # )
        if not keep_mask.all():
            targets = targets[keep_mask]  # [M,6], where M ≤ N
            masks = masks[keep_mask]  # ← now masks and gt_batch_idx both length M
            gt_batch_idx = gt_batch_idx[keep_mask]

            # if not self.overlap:
            #     masks = batch["masks"]  #    expects shape [sum_labels, H, W]
            #     # batch["masks"] = masks[keep_mask]
            #     gt_batch_idx = gt_batch_idx[keep_mask]

        num_after = targets.shape[0]
        num_masks_after = masks.shape[0]
        # print(
        #     f"DEBUG after  filtering: {num_after} targets, "
        #     f"{num_masks_after} masks kept"
        # )
        # -- END

        loss = torch.zeros(4, device=self.device)  # box, seg, cls, dfl

        if isinstance(preds, (list, tuple)) and len(preds) == 3:
            feats, pred_masks, proto = preds
        else:
            # e.g. when preds comes wrapped as (something, (feats, pred_masks, proto))
            feats, pred_masks, proto = preds[1]

        # print(
        #     f"DEBUG [Loss __call__] pred_masks mean={pred_masks.mean().item():.4f}, "
        #     f"std={pred_masks.std().item():.4f}"
        # )
        # print(
        #     f"DEBUG [Loss __call__] proto      mean={proto.mean().item():.4f}, "
        #     f"std={proto.std().item():.4f}"
        # )

        batch_size, _, mask_h, mask_w = (
            proto.shape
        )  # batch size, number of masks, mask height, mask width

        distri_and_score = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], dim=2
        )
        pred_distri, pred_scores = distri_and_score.split(
            (self.reg_max * 4, self.nc), dim=1
        )
        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        anchor_grid, stride_tensor = make_anchors(feats, self.stride, 0.5)
        anchor_points = anchor_grid.unsqueeze(0).expand(batch_size, -1, -1)

        pts = (anchor_points * stride_tensor).long()
        B, N, _ = pts.shape

        pts[..., 0].clamp_(0, ignore_map.shape[2] - 1)
        pts[..., 1].clamp_(0, ignore_map.shape[1] - 1)
        # batch_idx = torch.arange(B, device=self.device)[:, None]  # [B,1]
        anchor_batch_idx = torch.arange(B, device=self.device)[:, None]

        # --- ADDED: Remove class-0 GT objects before assignment ---
        # Ignore anchor: Purpose: Don't penalize predictions that fall on ignored regions.
        # anchor_keep = ignore_map[batch_idx, pts[..., 1], pts[..., 0]]  # [B,N]
        # anchor_keep = anchor_keep.bool()  # True = keep, False = ignore
        anchor_keep = ignore_map[anchor_batch_idx, pts[..., 1], pts[..., 0]].bool()

        # --- ADDED: Remove class-0 from loss/assignment ---
        # Targets
        try:
            targets = self.preprocess(
                targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
            )
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy

            mask_gt = (gt_bboxes.sum(2, keepdim=True) > 0.0) & (
                gt_labels != self.ignore_class
            )

            # print(f"DEBUG [Loss __call__] mask_gt sum={mask_gt.sum().item()}")

        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # print("DEBUG pred_bboxes:", pred_bboxes.shape)  # should be (B, N, 4)
        # print("DEBUG anchor_grid:", anchor_grid.shape)  # (N, 2)
        # print("DEBUG stride_tensor:", stride_tensor.shape)  # (N, 2)
        # print("DEBUG batched anchors:", anchor_points.shape)  # (B, N, 2)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            (anchor_grid * stride_tensor).to(self.device),
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # --- ADDED: Mask out ignore_class channel and ignored anchors in classification loss ---
        keep_mask = anchor_keep.unsqueeze(-1).expand(-1, -1, self.nc)  # [B, N, nc]
        target_scores = target_scores * keep_mask.to(target_scores.dtype)
        # Cls loss
        cls_loss_all = self.bce(pred_scores, target_scores)  # [B, N, nc]
        # Zero out the entire “ignore class” channel
        cls_loss_all[..., self.ignore_class] = 0
        #  Mask out any anchor that overlaps a class0 region entirely
        #    (anchor_keep is [B,N], expand to [B,N,nc])
        keep_mask = anchor_keep.unsqueeze(-1).float()
        cls_loss_all = cls_loss_all * keep_mask

        num_keep = keep_mask.sum() * (self.nc - 1)
        if num_keep.item() == 0:
            loss[2] = torch.tensor(
                0.0, device=self.device
            )  # no valid anchors → zero cls loss
        else:
            loss[2] = cls_loss_all.sum() / num_keep  # normal case

        # --- MODIFIED: Segmentation loss handles ignore pixels ---
        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = masks.to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            # --- ADDED: Upsample ignore_map to mask resolution and pass to segmentation loss ---
            # A: Add ignore_map handling in segmentation loss
            # upsample ignore_map to the mask resolution, keeping batch dimension
            # Result: [B, 1, Hm, Wm] -> squeeze channel -> [B, Hm, Wm]
            ignore_map_rs = F.interpolate(
                ignore_map.unsqueeze(1), (mask_h, mask_w), mode="nearest"
            ).squeeze(1)

            # DEBUG: check per-image keep counts
            # print(
            #     f"DEBUG [Loss __call__] ignore_map_rs shape={ignore_map_rs.shape}, "
            #     f"DEBUG batch sums={[ignore_map_rs[i].sum().item() for i in range(ignore_map_rs.shape[0])]}"
            # )

            # A: Segmentation loss: since we removed class 0 GTs, this only runs on class 1 instances
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                gt_batch_idx,
                proto,
                pred_masks,
                imgsz,
                self.overlap,
                ignore_map=ignore_map_rs,
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (
                pred_masks * 0
            ).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor,
        pred: torch.Tensor,
        proto: torch.Tensor,
        xyxy: torch.Tensor,
        area: torch.Tensor,
        ignore_crop: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        # --- MODIFIED: Single mask loss now supports ignore_crop ---
        min_valid_pixels = 10
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)

        # per-pixel BCE
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")

        # crop to each box
        cropped = crop_mask(loss, xyxy)  # (n, h, w)

        if ignore_crop is not None:
            ignore_crop = ignore_crop.float()
            cropped = cropped * ignore_crop
            keep_counts = ignore_crop.sum(dim=(1, 2)).clamp(
                min=min_valid_pixels
            )  # (n,)
            mean_per_obj = cropped.sum(dim=(1, 2)) / keep_counts  # (n,)
        else:
            mean_per_obj = cropped.mean(dim=(1, 2))  # (n,)

        # min_area = 100.0
        # area = area.clamp(min=min_area)
        # mean_per_obj = mean_per_obj / area

        # occasional debug logging kept intact
        # if torch.rand(1).item() < 0.01:
        # print(
        #     f"MASK_LOSS: mean_per_obj values: {mean_per_obj[:5].detach().cpu().tolist()}"
        # )
        # print(
        #     f"MASK_LOSS: total sum: {mean_per_obj.sum().item():.4f}, count: {len(mean_per_obj)}"
        # )

        # 5) mean over all positive instances
        return mean_per_obj.sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
        ignore_map: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, _, mask_h, mask_w = proto.shape

        # --- ensure ignore_map is valid ---
        if (
            ignore_map is None
            or ignore_map.ndim != 3
            or ignore_map.shape[0] != batch_size
        ):
            ignore_map = torch.ones(
                (batch_size, mask_h, mask_w),
                device=proto.device,
                dtype=proto.dtype,
            )

        # normalize & scale GT boxes to mask resolution
        tb_norm = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(tb_norm)[..., 2:].prod(2)
        mxyxy = tb_norm * torch.tensor(
            [mask_w, mask_h, mask_w, mask_h],
            device=proto.device,
            dtype=proto.dtype,
        )

        # --- build per-image mask stacks once ---
        # masks: [total_masks, H, W], batch_idx: [total_masks,1]
        flat_idx = batch_idx.view(-1)
        masks_by_image = [
            masks[flat_idx == i] for i in range(batch_size)  # -> [Ni, H, W]
        ]

        total_loss = 0.0
        total_instances = 0

        # zip everything except masks itself
        for i, (fg_i, idx_i, pm_i, proto_i, boxes_i, area_i) in enumerate(
            zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea)
        ):
            masks_i = masks_by_image[i]  # now shape [Ni, H, W]

            if not fg_i.any():
                # keep DDP happy
                total_loss += (proto_i * 0).sum() + (pm_i * 0).sum()
                continue

            # count positives and collect their GT indices
            n_instances = int(fg_i.sum().item())
            total_instances += n_instances
            mask_idx = idx_i[fg_i]  # shape [Mi]
            num_masks = masks_i.shape[0]

            if num_masks == 0:
                print(f"Image {i} has no GT masks!")
                continue

            # clamp into valid range
            clamped_idx = mask_idx.clamp(0, num_masks - 1)

            # build gt_mask of shape [Mi, H, W]
            if overlap:
                gt_mask = (masks_i == (clamped_idx + 1).view(-1, 1, 1)).float()
            else:
                gt_mask = masks_i[clamped_idx]

            # crop ignore_map under each positive box
            masks_for_crop = ignore_map[i].unsqueeze(0)  # [1, Hm, Wm]
            boxes_for_crop = boxes_i[fg_i]  # [Mi,4]
            ignore_crop = crop_mask(masks_for_crop, boxes_for_crop)

            # debug print original vs clamped
            # print(
            #     f"DEBUG mask selection: img={i}, Ni={num_masks}, "
            #     f"orig_idx={mask_idx.tolist()}, clamped={clamped_idx.tolist()}"
            # )

            # compute per-image mask loss
            img_loss = self.single_mask_loss(
                gt_mask,
                pm_i[fg_i],
                proto_i,
                boxes_i[fg_i],
                area_i[fg_i],
                ignore_crop=ignore_crop,
            )
            total_loss += img_loss

        # if total_instances == 0:
        #     return torch.tensor(0.0, device=proto.device)

        # # final normalization & safe print
        # num_pos = fg_mask.sum().clamp(min=1.0)
        # loss_val = total_loss.item() if hasattr(total_loss, "item") else total_loss
        # print(
        #     f"DEBUG total_seg_loss={loss_val:.4f} / pos={num_pos.item()}, instances={total_instances}"
        # )

        # return total_loss / total_instances
        # no positives → zero loss
        num_pos = fg_mask.sum().clamp(min=1.0)
        if num_pos == 0:
            return torch.tensor(0.0, device=proto.device)

        # print(f"DEBUG total_seg_loss={total_loss:.4f} / pos={num_pos.item()}")
        # average per‐anchor mask loss
        return total_loss / num_pos


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = (
            torch.from_numpy(OKS_SIGMA).to(self.device)
            if is_pose
            else torch.ones(nkpt, device=self.device) / nkpt
        )
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it for pose estimation."""
        # print("➡️  DEBUG Entering v8PoseLoss")

        loss = torch.zeros(
            5, device=self.device
        )  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(
            targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
        )
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        mask_gt = gt_labels == 1  # shape [B, N, 1]

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(
            anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape)
        )  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch_idx,
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
        
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self,
        masks,
        target_gt_idx,
        keypoints,
        batch_idx,
        stride_tensor,
        target_bboxes,
        pred_kpts,
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]),
            device=keypoints.device,
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1,
            target_gt_idx_expanded.expand(
                -1, -1, keypoints.shape[1], keypoints.shape[2]
            ),
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = (
                gt_kpt[..., 2] != 0
                if gt_kpt.shape[-1] == 3
                else torch.full_like(gt_kpt[..., 0], True)
            )
            kpts_loss = self.keypoint_loss(
                pred_kpt, gt_kpt, kpt_mask, area
            )  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(
                    pred_kpt[..., 2], kpt_mask.float()
                )  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        # print("➡️  DEBUG Entering v8ClassificationLoss")
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=10, num_classes=self.nc, alpha=0.5, beta=6.0
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for oriented bounding box detection."""
        # print("➡️ DEBUG Entering v8OBBLoss")

        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[
            0
        ]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = (
            torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype)
            * self.stride[0]
        )  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat(
                (batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1
            )
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[
                (rw >= 2) & (rh >= 2)
            ]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(
                targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
            )
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            # mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
            mask_gt = (gt_bboxes.sum(2, keepdim=True) > 0.0) & (
                gt_labels != self.ignore_class
            )
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(
            anchor_points, pred_distri, pred_angle
        )  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = (
            self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        )  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.type(pred_dist.dtype))
            )
        return torch.cat(
            (dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1
        )


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # print("DEBUG Entering E2EDetectLoss")
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model)
        # NOTE: store following info as it's changeable in __call__
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def __call__(self, preds, batch):
        """Calculate the loss for text-visual prompt detection."""
        # print("DEBUG Entering TVPDetectLoss")

        feats = preds[1] if isinstance(preds, tuple) else preds
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion(vp_feats, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, feats):
        """Extract visual-prompt features from the model output."""
        vnc = feats[0].shape[1] - self.ori_reg_max * 4 - self.ori_nc

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc

        return [
            torch.cat((box, cls_vp), dim=1)
            for box, _, cls_vp in [
                xi.split((self.ori_reg_max * 4, self.ori_nc, vnc), dim=1)
                for xi in feats
            ]
        ]


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model)

    def __call__(self, preds, batch):
        """Calculate the loss for text-visual prompt segmentation."""
        # print("DEBUG Entering TVPSegmentLoss")

        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        assert self.ori_reg_max == self.vp_criterion.reg_max  # TODO: remove it

        if self.ori_reg_max * 4 + self.ori_nc == feats[0].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        vp_feats = self._get_vp_features(feats)
        vp_loss = self.vp_criterion((vp_feats, pred_masks, proto), batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
