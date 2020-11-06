import numpy as np
import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric

from torchvision_detection_references.coco_eval import CocoEvaluator
from torchvision_detection_references.coco_utils import get_coco_api_from_dataset


class _CocoEvaluator(CocoEvaluator):
    def __init__(self, data_loader):
        self.coco = get_coco_api_from_dataset(data_loader.dataset)
        self.iou_type = "segm"  # _get_iou_types(model)

        super().__init__(self.coco, [self.iou_type])

        self.parameters = self.coco_eval[self.iou_type].params

    def calculate_average_precision(
        self, iou_threshold=None, area_range="all", n_detections_max=100
    ):
        coco_eval = self.coco_eval[self.iou_type]

        p = self.parameters

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_range]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == n_detections_max]

        # dimension of precision: [TxRxKxAxM]
        s = coco_eval.eval["precision"]
        # IoU
        if iou_threshold is not None:
            t = np.where(iou_threshold == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        return mean_s


class AveragePrecision(Metric):
    """
    Calculate coco evaluation.
    """

    def __init__(
        self,
        data_loader,
        device,
        iou_threshold=None,
        area_range="all",
        n_determinations_max=100,
    ):
        self._data_loader = data_loader
        self._device = device
        self._coco_evaluator = None
        self.iou_threshold = iou_threshold
        self.area_range = area_range
        self.n_determinations_max = n_determinations_max
        self._has_data = False
        self.parameters = None
        self.value = None

        super().__init__()

    def reset(self):
        self._has_data = False
        self._coco_evaluator = _CocoEvaluator(self._data_loader)
        self.parameters = self._coco_evaluator.parameters
        self.value = None

    def update(self, state):
        self._has_data = True
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)

        cpu_device = torch.device("cpu")

        outputs, targets = state

        targets = [{k: v.to(self._device) for k, v in t.items()} for t in targets]
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        self._coco_evaluator.update(res)
        torch.set_num_threads(n_threads)

    def compute(self):
        if not self._has_data:
            raise NotComputableError(
                "Average precision must have at least one sample before it can be computed."
            )

        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)

        self._coco_evaluator.synchronize_between_processes()
        self._coco_evaluator.accumulate()
        average_precision = self._coco_evaluator.calculate_average_precision(
            iou_threshold=self.iou_threshold,
            area_range=self.area_range,
            n_detections_max=self.n_determinations_max,
        )

        torch.set_num_threads(n_threads)

        self.value = average_precision

        return average_precision

    def print(self):
        p = self.parameters

        format_string = " {:<18} {} @[ IoU={:<9} | area={} | n_detections_max={:>3d} ] = {:0.3f}"
        title_string = "Average Precision"
        type_string = "(AP)"
        iou_string = (
            "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1])
            if self.iou_threshold is None
            else "{:0.2f}".format(self.iou_threshold)
        )

        print(
            format_string.format(
                title_string,
                type_string,
                iou_string,
                self.area_range,
                self.n_determinations_max,
                self.value,
            )
        )
