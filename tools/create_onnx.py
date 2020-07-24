import argparse
import logging
import os
import os.path as osp
import shutil
import tempfile

import torch
import torch.distributed as dist
from det3d import torchie
from det3d.core import coco_eval, results2json
from det3d.datasets import build_dataloader, build_dataset
from det3d.datasets.kitti import kitti_common as kitti
from det3d.datasets.kitti.eval import get_official_eval_result
from det3d.datasets.utils.kitti_object_eval_python.evaluate import (
    evaluate as kitti_evaluate,
)
from det3d.models import build_detector
from det3d.torchie.apis import init_dist
from det3d.torchie.apis.train import example_convert_to_torch
from det3d.torchie.parallel import MegDataParallel, MegDistributedDataParallel
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.trainer import example_to_device
from det3d.utils.dist.dist_common import (
    all_gather,
    get_rank,
    get_world_size,
    is_main_process,
    synchronize,
)
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="MegDet test detector")
    parser.add_argument("--config", help="test config file path",
        default='/workspace/Det3D/examples/point_pillars/configs/kitti_point_pillars_mghead_syncbn.py')
    parser.add_argument("--checkpoint", help="checkpoint file",
        default='/output/PointPillars_test_20200718-171357/latest.pth')
    parser.add_argument("--out", help="output result file")
    parser.add_argument(
        "--json_out", help="output result file name without extension", type=str
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        choices=["proposal", "proposal_fast", "bbox", "segm", "keypoints"],
        help="eval types",
    )
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--txt_result", action="store_true", help="save txt")
    parser.add_argument("--tmpdir", help="tmp dir for writing some results")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def create_onnx():
    # load model
    args = parse_args()

    # assert args.out or args.show or args.json_out, (
    #     "Please specify at least one operation (save or show the results) "
    #     'with the argument "--out" or "--show" or "--json_out"'
    # )

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    if args.json_out is not None and args.json_out.endswith(".json"):
        args.json_out = args.json_out[:-5]

    cfg = torchie.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # cfg.model.pretrained = None
    # cfg.data.test.test_mode = True
    cfg.data.val.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    # dataset = build_dataset(cfg.data.test)
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        # batch_size=cfg.data.samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        batch_size=1,
        workers_per_gpu=1,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    dbg = True
    if dbg:
        cfg.model.type = cfg.model.type + 'ListIOWrapper'
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    # model = MegDataParallel(model, device_ids=[0])
    model.to('cuda')

    # create some regular pytorch model...
    # model = alexnet(pretrained=True).eval().cuda()

    # create example data
    # x = torch.ones((1, 3, 224, 224)).cuda()
    model.eval()
    results_dict = []
    cpu_device = torch.device("cpu")

    results_dict = {}
    # prog_bar = torchie.ProgressBar(len(data_loader.dataset))
    dlit = iter(data_loader)
    batch = next(dlit)
    example = example_to_device(batch, device='cuda')

    with torch.no_grad():
        show = False
        if not dbg:
            outputs = model(example, return_loss=False, rescale=not show)
        else:
            outputs = model(example['voxels'].unsqueeze(0),
                            example['coordinates'].unsqueeze(0),
                            example['num_points'].unsqueeze(0),
                            example['num_voxels'].int().unsqueeze(0),
                            torch.tensor(example["shape"][0]).int().unsqueeze(0).to('cuda'),
                            example["anchors"][0].unsqueeze(0),
                            return_loss=False,
                            # no_nms=True
                            )


        # onnx
        input_shape_torch = torch.tensor(example["shape"][0])
        example_list = [example['voxels'].unsqueeze(0),
                        example['coordinates'].unsqueeze(0),
                        example['num_points'].unsqueeze(0),
                        example['num_voxels'].int().unsqueeze(0),
                        # input_shape_torch.int().unsqueeze(0).to('cuda'),
                        input_shape_torch.unsqueeze(0).to('cuda'),
                        example['anchors'][0].unsqueeze(0)]
        # input_names = ['voxels', 'coordinates', 'num_points', 'num_voxels', 'input_shape', 'anchors']
        input_names = ['voxels', 'coordinates', 'num_points', 'num_voxels', 'input_shape']
        # output_names = ['box3d_lidar', 'scores', 'label_preds']
        output_names = ['box_preds', 'cls_preds', 'dir_cls_preds']
        # TODO:
        # - compare results
        # - support dynamic input/output
        torch.onnx.export(model, tuple(example_list), 'pointpillars.onnx',
                          input_names=input_names, output_names=output_names,
                          dynamic_axes={'voxels': {1: 'nvoxel'},
                                        'coordinates': {1: 'nvoxel'},
                                        'num_points': {1: 'nvoxel'},
                                        'num_voxels': [],
                                        'input_shape': [],
                                        'anchors': []},
                          verbose=True,
                          opset_version=11,
                        #   opset_version=9,
                        #   opset_version=7,
                          )


    import onnxruntime as ort
    import numpy as np
    ort_session = ort.InferenceSession('pointpillars.onnx')
    outputs_onnx = ort_session.run(None, {'voxels': example_list[0].cpu().numpy(),
                                     'coordinates': example_list[1].cpu().numpy(),
                                     'num_points': example_list[2].cpu().numpy(),
                                     'num_voxels': example_list[3].cpu().numpy(),
                                     'input_shape': example_list[4].cpu().numpy(),
                                    #  'anchors': example_list[5].cpu().numpy()
                                    }
                                    )

    # torch script
    with torch.no_grad():
        # sm = torch.jit.script(model)

        model_trace = torch.jit.trace(model, example_list)
        model_trace.save('pointpillras_trace.pt')
        outputs_trace = model_trace(*example_list) 

    for i in range(len(outputs)):
        print('max abs error for output [{}]:'.format(i))
        print('onnx: {}'.format(np.abs(outputs_onnx[i] - outputs[i].cpu().numpy()).max()))
        print('trace: {}'.format(np.abs(outputs_trace[i].cpu().numpy() - outputs[i].cpu().numpy()).max()))

    # next sample
    batch = next(dlit)
    example = example_to_device(batch, device='cuda')
    with torch.no_grad():
        show = False
        if not dbg:
            outputs2 = model(example, return_loss=False, rescale=not show)
        else:
            outputs2 = model(example['voxels'].unsqueeze(0),
                            example['coordinates'].unsqueeze(0),
                            example['num_points'].unsqueeze(0),
                            example['num_voxels'].int().unsqueeze(0),
                            torch.tensor(example["shape"][0]).int().unsqueeze(0).to('cuda'),
                            example["anchors"][0].unsqueeze(0),
                            return_loss=False)

    example_list = [example['voxels'].unsqueeze(0),
                    example['coordinates'].unsqueeze(0),
                    example['num_points'].unsqueeze(0),
                    example['num_voxels'].int().unsqueeze(0),
                    # input_shape_torch.int().unsqueeze(0).to('cuda'),
                    input_shape_torch.unsqueeze(0).to('cuda'),
                    example['anchors'][0].unsqueeze(0),
                    ]

    outputs2_onnx = ort_session.run(None, {'voxels': example_list[0].cpu().numpy(),
                                     'coordinates': example_list[1].cpu().numpy(),
                                     'num_points': example_list[2].cpu().numpy(),
                                     'num_voxels': example_list[3].cpu().numpy(),
                                     'input_shape': example_list[4].cpu().numpy(),
                                    #  'anchors': example_list[5].cpu().numpy()
                                    }
                                    )

    with torch.no_grad():
        outputs2_trace = model_trace(*example_list) 

    # print(outputs2[1])
    # print(outputs2_onnx[1])
    # print(outputs_onnx[1] - outputs[1].cpu().numpy())

    for i in range(len(outputs2)):
        print('max abs error for output [{}]:'.format(i))
        print('onnx: {}'.format(np.abs(outputs2_onnx[i] - outputs2[i].cpu().numpy()).max()))
        print('trace: {}'.format(np.abs(outputs2_trace[i].cpu().numpy() - outputs2[i].cpu().numpy()).max()))


if __name__ == "__main__":
    # main()
    create_onnx()
