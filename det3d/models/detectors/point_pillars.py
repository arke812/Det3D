from typing import Dict
import torch
from torch import Tensor

from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        # self._with_neck = self.with_neck
        self._no_nms = False

    def no_nms(self, no_nms: bool=True):
        self._no_nms = no_nms

    def extract_feat(self, data: Dict[str, Tensor]):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example: Dict[str, Tensor], return_loss: bool=True):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = torch.tensor(num_voxels.shape[0])

        data: Dict[str, Tensor] = {}
        data['features'] = voxels
        data['num_voxels'] = num_points_in_voxel
        data['coors'] = coordinates
        data['batch_size'] = batch_size
        data['input_shape'] = example["shape"][0]

        x = self.extract_feat(data)
        preds = self.bbox_head(x)

        if self._no_nms:
            return preds

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)


@DETECTORS.register_module
class PointPillarsListIOWrapper(PointPillars):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super().__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )
        # self.pp = PointPillars(reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained)

    def forward(self,
        voxels,
        coordinates,
        num_points_in_voxel,
        num_voxels,
        input_shape,
        anchors,
        return_loss: bool=False,
        # **kwargs
        ):
        """
        """
        # no_nms: bool = False

        example: Dict[str, Tensor] = {}
        example['voxels'] = voxels.squeeze(0)
        example['coordinates'] = coordinates.squeeze(0)
        example['num_points'] = num_points_in_voxel.squeeze(0)
        example['num_voxels'] = num_voxels.squeeze(0)
        example['shape'] = input_shape
        example['anchors'] = anchors

        # out_dict = self.pp.forward(
        out_dict = super().forward(
            example,
            return_loss,
        )

        if self._no_nms == True:
            out_list = [(o['box_preds'], o['cls_preds'], o['dir_cls_preds']) for o in out_dict]
        else:
            out_list = [(o['box3d_lidar'], o['scores'], o['label_preds']) for o in out_dict]

        if len(out_list) != 1:
            RuntimeError('batch size must be 1.')

        return out_list[0]




