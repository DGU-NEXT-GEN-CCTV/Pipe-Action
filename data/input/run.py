import os
from argparse import ArgumentParser
from typing import Dict

from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases

filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input-dir', type=str, default="data/input")
    parser.add_argument('--vis-out-dir', type=str, default='data/output/vis')
    parser.add_argument('--pred-out-dir', type=str, default='data/output/pose')
    parser.add_argument('--pose2d', type=str, default=None)
    parser.add_argument('--pose2d-weights', type=str, default=None)
    parser.add_argument('--pose3d', type=str, default=None)
    parser.add_argument('--pose3d-weights', type=str, default=None)
    parser.add_argument('--det-model', type=str, default=None)
    parser.add_argument('--det-weights', type=str, default=None)
    parser.add_argument('--det-cat-ids', type=int, nargs='+', default=0)
    parser.add_argument('--scope', type=str, default='mmpose')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--show-progress', type=bool, default=True)
    args, _ = parser.parse_known_args()
    for model in POSE2D_SPECIFIC_ARGS:
        if args.pose2d is not None and model in args.pose2d:
            filter_args.update(POSE2D_SPECIFIC_ARGS[model])
            break 
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--draw-bbox', action='store_true')
    parser.add_argument('--draw-heatmap', action='store_true', default=False)
    parser.add_argument('--bbox-thr', type=float, default=filter_args['bbox_thr'])
    parser.add_argument('--nms-thr', type=float, default=filter_args['nms_thr'])
    parser.add_argument('--pose-based-nms', type=lambda arg: arg.lower() in ('true', 'yes', 't', 'y', '1'), default=filter_args['pose_based_nms'])
    parser.add_argument('--kpt-thr', type=float, default=0.3)
    parser.add_argument('--tracking-thr', type=float, default=0.3)
    parser.add_argument('--use-oks-tracking', action='store_true')
    parser.add_argument('--disable-norm-pose-2d', action='store_true')
    parser.add_argument('--disable-rebase-keypoint', action='store_true', default=False)
    parser.add_argument('--num-instances', type=int, default=1)
    parser.add_argument('--radius', type=int, default=3)
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--skeleton-style', default='mmpose', type=str, choices=['mmpose', 'openpose'])
    parser.add_argument('--black-background', action='store_true')
    parser.add_argument('--show-alias', action='store_true')

    call_args = vars(parser.parse_args())

    init_kws = [
        'pose2d', 'pose2d_weights', 'scope', 'device', 'det_model', 'det_weights', 'det_cat_ids', 'pose3d', 'pose3d_weights', 'show_progress'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    display_alias = call_args.pop('show_alias')

    return init_args, call_args, display_alias


def display_model_aliases(model_aliases: Dict[str, str]) -> None:
    aliases = list(model_aliases.keys())
    max_alias_length = max(map(len, aliases))
    print(f'{"ALIAS".ljust(max_alias_length+2)}MODEL_NAME')
    for alias in sorted(aliases):
        print(f'{alias.ljust(max_alias_length+2)}{model_aliases[alias]}')
        
    
def load_video_list(input_dir: str) -> list:
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    video_files = sorted(video_files)
    if not video_files:
        raise ValueError(f"No video files found in the input directory '{input_dir}'.")
    
    return [os.path.join(input_dir, f) for f in video_files]


def main():
    init_args, call_args, display_alias = parse_args()
    print(call_args['input_dir'])
    video_list = load_video_list(call_args['input_dir'])
    
    inferencer = MMPoseInferencer(**init_args)
    
    for video_path in video_list:
        cur_call_args = call_args.copy()
        cur_call_args['inputs'] = video_path
        for _ in inferencer(**call_args):
            pass

if __name__ == '__main__':
    main()
