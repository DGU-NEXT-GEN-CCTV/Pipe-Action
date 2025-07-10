import os
import cv2
import rich
from rich.table import Table
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Dict

from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases      

console = rich.get_console()
console = console.__class__(log_time=False)      

filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
)

def console_banner():
    console.clear()
    banner = [
        "\n",
        "    ___       ___       ___       ___            ___       ___       ___            ___       ___       ___       ___    ",
        "   /\__\     /\  \     /\__\     /\  \          /\  \     /\  \     /\__\          /\  \     /\  \     /\  \     /\__\   ",
        "  /:| _|_   /::\  \   |::L__L    \:\  \        /::\  \   /::\  \   /:| _|_        /::\  \   /::\  \    \:\  \   /:/ _/_  ",
        " /::|/\__\ /::\:\__\ /::::\__\   /::\__\      /:/\:\__\ /::\:\__\ /::|/\__\      /:/\:\__\ /:/\:\__\   /::\__\ |::L/\__\ ",
        " \/|::/  / \:\:\/  / \;::;/__/  /:/\/__/      \:\:\/__/ \:\:\/  / \/|::/  /      \:\ \/__/ \:\ \/__/  /:/\/__/ |::::/  / ",
        "   |:/  /   \:\/  /   |::|__|   \/__/          \::/  /   \:\/  /    |:/  /        \:\__\    \:\__\    \/__/     L;;/__/  ",
        "   \/__/     \/__/     \/__/                    \/__/     \/__/     \/__/          \/__/     \/__/            ",
        "\n",
        "Pipe-Action: Pipeline for Preprocessing Video Datasets for Action Recognition Model Training (ProtoGCN)",
        "\n",
    ]
    for line in banner:
        console.print(line, style="bold green")
        
def console_args(args):
    table = Table(show_header=True, show_footer=False)
    table.add_column("Argument", width=17)
    table.add_column("Value", width=40)
    for arg, value in args.items():
        table.add_row(arg, str(value))
    console.print(table)

def parse_args():
    console.log("[bold] ‣ Initializing... [/bold]")
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
    parser.add_argument('--show-progress', action='store_true')
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
    
    console.log(f"[bold green] ∙ Init Arguments[/bold green]")
    console_args(init_args)
    console.log(f"[bold green] ∙ Call Arguments[/bold green]")
    console_args(call_args)

    return init_args, call_args, display_alias


def display_model_aliases(model_aliases: Dict[str, str]) -> None:
    aliases = list(model_aliases.keys())
    max_alias_length = max(map(len, aliases))
    print(f'{"ALIAS".ljust(max_alias_length+2)}MODEL_NAME')
    for alias in sorted(aliases):
        print(f'{alias.ljust(max_alias_length+2)}{model_aliases[alias]}')
        
    
def load_video_list(input_dir: str) -> list:
    console.log("[bold] ‣ Loading Videos... [/bold]")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")
    
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    video_files = sorted(video_files)
    if not video_files:
        raise ValueError(f"No video files found in the input directory '{input_dir}'.")
    
    console.log(f"[bold green] ∙ Found {len(video_files)} videos in '{input_dir}'[/bold green]")
    
    return [os.path.join(input_dir, f) for f in video_files]

def load_model(init_args: Dict[str, str]) -> MMPoseInferencer:
    console.log("[bold] ‣ Loading Model... [/bold]")
    inferencer = MMPoseInferencer(**init_args)
    console.log(f"[bold green] ∙ Model loaded[/bold green]")
    
    return inferencer

def main():
    console_banner() # Setting Progress
    init_args, call_args, display_alias = parse_args()    
    video_list = load_video_list(call_args['input_dir'])
    video_num = len(video_list)
    inferencer = load_model(init_args)
    
    console_banner() # Inferencing Progress
    console.log("[bold] ‣ Inferencing... [/bold]")
    for video_idx, video_path in enumerate(video_list):
        cur_call_args = call_args.copy()
        cur_call_args['inputs'] = video_path
        total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(inferencer(**cur_call_args), total=total_frames, desc=f'[{video_idx + 1:04d}/{video_num:04d}] Processing {os.path.basename(video_path)}'):
            pass

    console.log("[bold green] ∙ Inferencing completed [/bold green]")

if __name__ == '__main__':
    main()
