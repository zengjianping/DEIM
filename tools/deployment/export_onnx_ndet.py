"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from engine.core import YAMLConfig


def main(args):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        state.pop('decoder.anchors')
        state.pop('decoder.valid_mask')
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, max_dets, thres_conf) -> None:
            super().__init__()
            self.max_dets = max_dets
            self.thres_conf = thres_conf
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images):
            outputs = self.model(images)

            image_size = torch.tensor(images.shape[2:4])
            outputs = self.postprocessor(outputs, image_size)
            labels, bboxes, scores = outputs

            max_dets = min(labels.shape[1], self.max_dets)
            scores, index = torch.topk(scores, max_dets, dim=-1)
            labels = torch.gather(labels, dim=1, index=index)
            bboxes = torch.gather(bboxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, bboxes.shape[-1]))
            num_dets = torch.ge(scores, self.thres_conf).sum(-1, True)
            
            labels = labels.to(torch.int32)
            num_dets = num_dets.to(torch.int32)

            return num_dets, bboxes, scores, labels

    model = Model(args.max_dets, args.thres_conf)
    data = torch.rand(1, 3, 640, 640)
    _ = model(data)

    dynamic_axes = {
        "images": {0: "batch"},
        "num_dets": {0: "batch"},
        "det_boxes": {0: "batch"},
        "det_scores": {0: "batch"},
        "det_classes": {0: "batch"},
    }

    output_file = args.resume.replace('.pth', '.onnx') if args.resume else 'model.onnx'

    torch.onnx.export(
        model,
        (data,),
        output_file,
        input_names=['images'],
        output_names=["num_dets", 'det_boxes', 'det_scores', 'det_classes'],
        dynamic_axes=dynamic_axes,
        opset_version=18,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim
        dynamic = True
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {'images': data.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/dfine/dfine_hgnetv2_l_coco.yml', type=str, )
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--check',  action='store_true', default=False)
    parser.add_argument('--simplify',  action='store_true', default=False)
    parser.add_argument('--max_dets', type=int, default=100)
    parser.add_argument('--thres_conf', type=float, default=0.3)
    args = parser.parse_args()
    main(args)
