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
    import torch
    import torch2trt as t2t
    from torch2trt import torch2trt

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
    batch = next(iter(data_loader))
    # example = example_convert_to_torch(batch, device=device)
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
                            return_loss=False)
        # for output in outputs:
        #     token = output["metadata"]["token"]
        #     for k, v in output.items():
        #         if k not in [
        #             "metadata",
        #         ]:
        #             output[k] = v.to(cpu_device)
        #     results_dict.update(
        #         {token: output,}
        #     )
    # return results_dict




    # convert to TensorRT feeding sample data as input
    # from torchvision.models.alexnet import alexnet
    # model = alexnet(pretrained=False).eval().cuda()
    # x = torch.ones((1, 3, 224, 224)).cuda()
    # model_trt = torch2trt(model, [x])

    # memo: first dimension must be a batch dimension for torch2trt...

        input_shape_torch = torch.tensor(example["shape"][0])
        example_list = [example['voxels'].unsqueeze(0),
                        example['coordinates'].unsqueeze(0),
                        example['num_points'].unsqueeze(0),
                        example['num_voxels'].int().unsqueeze(0),
                        # input_shape_torch.int().unsqueeze(0).to('cuda'),
                        input_shape_torch.unsqueeze(0).to('cuda'),
                        example['anchors'][0].unsqueeze(0)]
        input_names = ['voxels', 'coordinates', 'num_points', 'num_voxels', 'input_shape', 'anchors']
        output_names = ['box3d_lidar', 'scores', 'label_preds']
        # TODO:
        # - compare results
        # - support dynamic input/output
        torch.onnx.export(model, tuple(example_list), 'pointpillars.onnx',
                          input_names=input_names, output_names=output_names,
                          verbose=True, opset_version=11)

        # input_features = model.reader(
        #     example['voxels'], example['num_voxels'].int(), example['coordinates']
        # )
        # tmp_mdl = PointPillarsScatterTest()
        # test_input = (input_features, example['coordinates'], torch.tensor(1).to('cuda'), input_shape_torch.to('cuda'))
        # torch.onnx.export(tmp_mdl, #model.backbone,
        #                   test_input,
        #                   'pointpillars.onnx', verbose=True, opset_version=11)

    # # caffe2
    # import onnx

    # # Load the ONNX model
    # model = onnx.load('pointpillars.onnx')

    # # Check that the IR is well formed
    # onnx.checker.check_model(model)

    # # Print a human readable representation of the graph
    # onnx.helper.printable_graph(model.graph)
    # import caffe2.python.onnx.backend as backend
    # import numpy as np
    # rep = backend.prepare(model, device="CUDA:0") # or "CPU"
    # # For the Caffe2 backend:
    # #     rep.predict_net is the Caffe2 protobuf for the network
    # #     rep.workspace is the Caffe2 workspace for the network
    # #       (see the class caffe2.python.onnx.backend.Workspace)
    # outputs = rep.run(np.random.randn(10, 3, 224, 224).astype(np.float32))
    # # To run networks with more than one input, pass a tuple
    # # rather than a single numpy ndarray.
    # print(outputs[0])



    import onnxruntime as ort
    import numpy as np
    ort_session = ort.InferenceSession('pointpillars.onnx')
    outputs_onnx = ort_session.run(None, {'voxels': example_list[0].cpu().numpy(),
                                     'coordinates': example_list[1].cpu().numpy(),
                                     'num_points': example_list[2].cpu().numpy(),
                                     'num_voxels': example_list[3].cpu().numpy(),
                                     'input_shape': example_list[4].cpu().numpy(),
                                     'anchors': example_list[5].cpu().numpy()
                                    }
                                    )

    print(outputs[1])
    print(outputs_onnx[1])


if __name__ == "__main__":
    # main()
    create_onnx()
