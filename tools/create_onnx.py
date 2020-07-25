import argparse

import torch
from det3d import torchie
from det3d.datasets import build_dataloader, build_dataset

from det3d.models import build_detector
from det3d.torchie.apis.train import example_convert_to_torch
from det3d.torchie.trainer import load_checkpoint
from det3d.torchie.trainer.trainer import example_to_device

import onnxruntime as ort
ort.set_default_logger_severity(0)
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description="MegDet test detector")
    parser.add_argument("--config", help="test config file path",
        default='/workspace/Det3D/examples/point_pillars/configs/kitti_point_pillars_mghead_syncbn.py')
    parser.add_argument("--checkpoint", help="checkpoint file",
        default='/output/PointPillars_test_20200718-171357/latest.pth')
    args = parser.parse_args()
    return args


def create_onnx():
    # load model
    args = parse_args()

    cfg = torchie.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.data.val.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    distributed = False

    # build the dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        batch_size=1,
        workers_per_gpu=1,
        dist=distributed,
        shuffle=False,
    )

    # build the model and load checkpoint
    no_nms = True

    # original model
    model_org = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model_org, args.checkpoint, map_location="cpu")
    model_org.to('cuda')
    model_org.no_nms(no_nms)
    model_org.eval()

    # model for export
    cfg.model.type = cfg.model.type + 'ListIOWrapper'
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to('cuda')
    model.no_nms(no_nms)
    model.eval()

    results_dict = []
    cpu_device = torch.device("cpu")

    dlit = iter(data_loader)
    batch = next(dlit)
    example = example_to_device(batch, device='cuda')
    example_list = [example['voxels'].unsqueeze(0),
                    example['coordinates'].unsqueeze(0),
                    example['num_points'].unsqueeze(0),
                    example['num_voxels'].int().unsqueeze(0),
                    torch.tensor(example["shape"][0]).int().unsqueeze(0).to('cuda'),
                    example['anchors'][0].unsqueeze(0)]

    with torch.no_grad():
        # test inference
        outputs_org = model_org(example, return_loss=False)
        outputs_org = [outputs_org[0]['box_preds'], outputs_org[0]['cls_preds'], outputs_org[0]['dir_cls_preds']]
        outputs = model(*example_list, return_loss=False)

        # onnx
        input_names = ['voxels', 'coordinates', 'num_points', 'num_voxels', 'input_shape']
        output_names = ['box_preds', 'cls_preds', 'dir_cls_preds']
        torch.onnx.export(model, tuple(example_list), 'pointpillars.onnx',
                          input_names=input_names, output_names=output_names,
                          dynamic_axes={'voxels': {1: 'nvoxel'},
                                        'coordinates': {1: 'nvoxel'},
                                        'num_points': {1: 'nvoxel'}},
                          verbose=True,
                          opset_version=11,
                          do_constant_folding=True,
                          )

    # onnx test inference
    ort_session = ort.InferenceSession('pointpillars.onnx')
    outputs_onnx = ort_session.run(None, {'voxels': example_list[0].cpu().numpy(),
                                     'coordinates': example_list[1].cpu().numpy(),
                                     'num_points': example_list[2].cpu().numpy(),
                                     'input_shape': example_list[4].cpu().numpy(),
                                    }
                                    )

    # torch script
    with torch.no_grad():
        # sm = torch.jit.script(model) # too much work required to make this work

        model_trace_ = torch.jit.trace(model, example_list)
        model_trace_.save('pointpillars_trace.pt')
        model_trace = torch.jit.load('pointpillars_trace.pt')
        outputs_trace = model_trace(*example_list) 

    for i in range(len(outputs)):
        print('max abs error for output [{}]:'.format(i))
        print('trace: {}'.format(np.abs(outputs_trace[i].cpu().numpy() - outputs[i].cpu().numpy()).max()))
        print('org: {}'.format(np.abs(outputs_org[i].cpu().numpy() - outputs[i].cpu().numpy()).max()))
        print('onnx: {}'.format(np.abs(outputs_onnx[i] - outputs[i].cpu().numpy()).max()))

    # next samples
    for i_b in range(5):
        batch = next(dlit)
        example = example_to_device(batch, device='cuda')
        example_list = [example['voxels'].unsqueeze(0),
                        example['coordinates'].unsqueeze(0),
                        example['num_points'].unsqueeze(0),
                        example['num_voxels'].int().unsqueeze(0),
                        torch.tensor(example["shape"][0]).int().unsqueeze(0).to('cuda'),
                        example["anchors"][0].unsqueeze(0)]

        # torch
        with torch.no_grad():
            outputs_org = model_org(example, return_loss=False)
            outputs_org = [outputs_org[0]['box_preds'], outputs_org[0]['cls_preds'], outputs_org[0]['dir_cls_preds']]
            outputs = model(*example_list, return_loss=False)

        # onnx
        onnx_input = {'voxels': example_list[0].cpu().numpy(),
                    'coordinates': example_list[1].cpu().numpy(),
                    'num_points': example_list[2].cpu().numpy(),
                    'input_shape': example_list[4].cpu().numpy(),
                    }
        outputs_onnx = ort_session.run(None, onnx_input)

        # torch jit
        with torch.no_grad():
            outputs_trace = model_trace(*example_list) 

        # error check
        for i in range(len(outputs)):
            print('max abs error for output [{}] at {}-th sample:'.format(i, i_b))
            print('onnx: {}'.format(np.abs(outputs_onnx[i] - outputs[i].cpu().numpy()).max()))
            print('trace: {}'.format(np.abs(outputs_trace[i].cpu().numpy() - outputs[i].cpu().numpy()).max()))
            print('org: {}'.format(np.abs(outputs_org[i].cpu().numpy() - outputs[i].cpu().numpy()).max()))

        # performance test
        from timeit import timeit
        N = 10
        with torch.no_grad():
            print('computation time:')
            # print('org_with_nms: {} s'.format(timeit(lambda: model_with_nms(example, return_loss=False), number=N)/N))
            print('org: {} s'.format(timeit(lambda: model(*example_list, return_loss=False), number=N)/N))
            print('trace: {} s'.format(timeit(lambda: model_trace(*example_list), number=N)/N))
            try: # sometimes onnx raise error. why?
                print('onnx: {} s'.format(timeit(lambda: ort_session.run(None, onnx_input), number=N)/N))
            except:
                pass


if __name__ == "__main__":
    # main()
    create_onnx()
