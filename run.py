import os
import rich
from rich.table import Table
import cv2
import json
import pickle
import random
import numpy as np
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
    parser.add_argument('--output-dir', type=str, default="data/output")
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


def convert_annotation(video_path: str, pose_dir: str) -> list:
    def convert_relative_coord(keypoints: list): # 상대 좌표값으로 변환
        processed_keypoints = []
        
        for k in keypoints:
            base_j = ((k[11][0] + k[12][0]) / 2, (k[11][1] + k[12][1]) / 2) # 기준점 (골반 중심점, COCO 17 keypoints: 11 = right hip, 12 = left hip)
            base_l = np.linalg.norm(np.array(k[11]) - np.array(k[12])) # 기준 길이 (골반 중심점과 양쪽 엉덩이 사이의 거리)
            processed_k = []
            for j in k:
                processed_k.append([(j[0] - base_j[0]) / base_l, (j[1] - base_j[1]) / base_l]) # 상대 좌표로 변환된 키포인트에 대한 일반화
            processed_keypoints.append(processed_k)
            
        return processed_keypoints
    
    def load_pose(pose_path: str):
        if not os.path.exists(pose_path):
            return [], []
        
        with open(pose_path, 'r') as f:
            data = json.load(f)
            
        max_person_num = 6
        keypoints = [[] for _ in range(max_person_num)] # 최대 6명의 사람에 대한 키포인트
        keypoint_scores = [[] for _ in range(max_person_num)] # 최대 6명의 사람에 대한 키포인트 정확도 점수

        for frame in data:
            pose = frame['instances']
            k = [instance['keypoints'] for instance in pose]
            k_r = convert_relative_coord(k)
            k_s = [instance['keypoint_scores'] for instance in pose]
            for i, c_k_r in enumerate(k_r):
                if i < max_person_num:
                    keypoints[i].append(c_k_r) # 상대 좌표로 변환된 키포인트를 추가
            for i, c_k_s in enumerate(k_s):
                if i < max_person_num:
                    keypoint_scores[i].append(c_k_s)
                    
        processed_keypoints = []
        processed_keypoint_scores = []
                    
        for p_k in keypoints:
            if len(p_k) > 0:
                processed_keypoints.append(np.array(p_k, dtype=np.float16))
        for p_k_s in keypoint_scores:
            if len(p_k_s) > 0:
                processed_keypoint_scores.append(np.array(p_k_s, dtype=np.float16))

        return processed_keypoints, processed_keypoint_scores
    
    annotation = {
        'frame_dir': '', # 동영상 이름
        'label': '', # 레이블
        'img_shape': (0, 0), # 이미지 크기
        'original_shape': (0, 0), # 원본 크기
        'total_frames': 0, # 총 프레임 수
        'keypoint': [], # 키포인트
        'keypoint_score': [] # 키포인트 정확도 점수
    }
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video = cv2.VideoCapture(video_path)
    resolution = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    annotation['frame_dir'] = video_name
    annotation['img_shape'] = resolution
    annotation['original_shape'] = resolution
    annotation['total_frames'] = total_frames
    
    keypoints, keypoint_scores = load_pose(os.path.join(pose_dir, f'{video_name}.json'))
    
    annotations = []
    
    for k, k_s in zip(keypoints, keypoint_scores):
        if len(k) > 0:
            a = annotation.copy()
            a['keypoint'] = (np.array([k], dtype=np.float16))
            a['keypoint_score'] = (np.array([k_s], dtype=np.float16))
            annotations.append(a)

    return annotations


def compress_dataset(video_list: list, pose_dir: str, output_dir: str, model: str) -> str:
    dataset = {
        'split': {
            'train': [],
            'val': [],
        },
        'annotations': []
    }
    
    for video_path in video_list: # 동영상을 train: 8/val: 2로 분할
        if random.random() < 0.8:
            dataset['split']['train'].append(video_path)
        else:
            dataset['split']['val'].append(video_path)

    for video_path in video_list:
        annotations = convert_annotation(video_path, pose_dir)
        dataset['annotations'].extend(annotations)
    
    dataset_path = os.path.join(output_dir, f'dataset_{model}.pkl')
    pickle.dump(dataset, open(dataset_path, 'wb'))
    
    return dataset_path

def main():
    console_banner() # Setting Progress
    init_args, call_args, display_alias = parse_args()    
    video_list = load_video_list(call_args['input_dir'])
    video_num = len(video_list)
    inferencer = load_model(init_args)
    os.makedirs(call_args['output_dir'], exist_ok=True)
    os.makedirs(call_args['pred_out_dir'], exist_ok=True)
    os.makedirs(call_args['vis_out_dir'], exist_ok=True)
    
    console_banner() # Inferencing Progress
    console.log("[bold] ‣ Inferencing... [/bold]")
    
    for video_idx, video_path in enumerate(video_list):
        cur_call_args = call_args.copy()
        cur_call_args['inputs'] = video_path
        total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(inferencer(**cur_call_args), total=total_frames, desc=f'[{video_idx + 1:04d}/{video_num:04d}] Processing {os.path.basename(video_path)}'):
            pass

    console.log("[bold green] ∙ Inferencing completed [/bold green]")
    
    console_banner() # Compressing Progress
    console.log("[bold] ‣ Compressing... [/bold]")
    dataset = compress_dataset(video_list, call_args['pred_out_dir'], call_args['output_dir'], init_args['pose2d'])
    console.log(f"[bold green] ∙ Compressing completed, {dataset} [/bold green]\n")

if __name__ == '__main__':
    main()
