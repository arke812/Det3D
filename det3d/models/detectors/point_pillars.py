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

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = num_voxels.shape[0]

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds).update(preds)
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
        super(PointPillarsListIOWrapper, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def forward(self,
        voxels,
        coordinates, # should be const?
        num_points_in_voxel,
        num_voxels,
        input_shape, # should be const
        anchors,
        return_loss=False,
        # **kwargs
        ):
        """
        limitation for tensorrt convertion:
         - first dimension must be batch dimension. this dim size is overwritten by 1
        """
        example = dict(
            voxels=voxels.squeeze(0),
            coordinates=coordinates.squeeze(0),
            num_points=num_points_in_voxel.squeeze(0),
            num_voxels=num_voxels.squeeze(0),
            shape=input_shape,
            anchors=anchors,
        )
        out_dict = super(PointPillarsListIOWrapper, self).forward(
            example,
            return_loss,
            # **kwargs
        )

        out_list = [(o['box3d_lidar'], o['scores'], o['label_preds']) for o in out_dict]
        # out_list = [(o['box_preds'], o['cls_preds'], o['dir_cls_preds']) for o in out_dict]

        if len(out_list) != 1:
            RuntimeError('batch size must be 1.')

        return out_list[0]




