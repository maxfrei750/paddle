from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from .roi_heads import RoIHeads


class FiberDimensionPredictor(nn.Module):
    """
    Fiber dimension (width or length) prediction layers for FibeR-CNN

    :param in_channels: number of input channels
    """

    def __init__(self, in_channels: int):

        super().__init__()
        self.fiber_dimension = nn.Linear(in_channels, 1)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        return self.fiber_dimension(x)


class FibeRCNN(FasterRCNN):
    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        # Mask parameters
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        # keypoint parameters
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        num_keypoints=17,
        # fiber width parameters
        fiber_width_roi_pool=None,
        fiber_width_head=None,
        fiber_width_predictor=None,
        # fiber length parameters
        fiber_length_roi_pool=None,
        fiber_length_head=None,
        fiber_length_predictor=None,
    ):
        out_channels = backbone.out_channels

        mask_head, mask_predictor, mask_roi_pool = self._get_mask_branch_parts(
            mask_head, mask_predictor, mask_roi_pool, num_classes, out_channels
        )

        keypoint_head, keypoint_predictor, keypoint_roi_pool = self._get_keypoint_branch_parts(
            keypoint_head,
            keypoint_predictor,
            keypoint_roi_pool,
            num_classes,
            num_keypoints,
            out_channels,
        )

        (
            fiber_width_head,
            fiber_width_predictor,
            fiber_width_roi_pool,
        ) = self._get_fiber_width_branch_parts(
            fiber_width_head, fiber_width_predictor, fiber_width_roi_pool, out_channels
        )

        (
            fiber_length_head,
            fiber_length_predictor,
            fiber_length_roi_pool,
        ) = self._get_fiber_length_branch_parts(
            fiber_length_head, fiber_length_predictor, fiber_length_roi_pool, out_channels
        )

        super().__init__(
            backbone,
            num_classes,
            min_size,
            max_size,
            image_mean,
            image_std,
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
        )

        self.roi_heads = RoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            # Mask
            mask_roi_pool,
            mask_head,
            mask_predictor,
            # Keypoints
            keypoint_roi_pool,
            keypoint_head,
            keypoint_predictor,
            # Fiber Width
            fiber_width_roi_pool,
            fiber_width_head,
            fiber_width_predictor,
            # Fiber Length
            fiber_length_roi_pool,
            fiber_length_head,
            fiber_length_predictor,
        )

    @staticmethod
    def _get_fiber_dimension_branch_parts(
        fiber_dimension_head, fiber_dimension_predictor, fiber_dimension_roi_pool, out_channels
    ):
        assert isinstance(fiber_dimension_roi_pool, (MultiScaleRoIAlign, type(None)))

        if fiber_dimension_roi_pool is None:
            fiber_dimension_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
            )
        if fiber_dimension_head is None:
            resolution = fiber_dimension_roi_pool.output_size[0]
            representation_size = 1024
            fiber_dimension_head = TwoMLPHead(out_channels * resolution ** 2, representation_size)
        if fiber_dimension_predictor is None:
            representation_size = 1024
            fiber_dimension_predictor = FiberDimensionPredictor(representation_size)
        return fiber_dimension_head, fiber_dimension_predictor, fiber_dimension_roi_pool

    @staticmethod
    def _get_keypoint_branch_parts(
        keypoint_head,
        keypoint_predictor,
        keypoint_roi_pool,
        num_classes,
        num_keypoints,
        out_channels,
    ):
        assert isinstance(keypoint_roi_pool, (MultiScaleRoIAlign, type(None)))
        if num_classes is not None:
            if keypoint_predictor is not None:
                raise ValueError("num_classes should be None when keypoint_predictor is specified")
        if keypoint_roi_pool is None:
            keypoint_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
            )
        if keypoint_head is None:
            keypoint_layers = tuple(512 for _ in range(8))
            keypoint_head = KeypointRCNNHeads(out_channels, keypoint_layers)
        if keypoint_predictor is None:
            keypoint_dim_reduced = 512  # == keypoint_layers[-1]
            keypoint_predictor = KeypointRCNNPredictor(keypoint_dim_reduced, num_keypoints)
        return keypoint_head, keypoint_predictor, keypoint_roi_pool

    @staticmethod
    def _get_mask_branch_parts(mask_head, mask_predictor, mask_roi_pool, num_classes, out_channels):
        assert isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None)))
        if num_classes is not None:
            if mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")
        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(
                featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2
            )
        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)
        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(
                mask_predictor_in_channels, mask_dim_reduced, num_classes
            )
        return mask_head, mask_predictor, mask_roi_pool
