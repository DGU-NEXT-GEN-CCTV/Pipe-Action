![thumbnail](./thumb.png)

# Pipeline Action

이 저장소는 행동 추정 모델을 학습시키기 위한 데이터셋을 구축하는 모듈을 제공합니다.

### Note

모든 테스트는 다음 환경에서 진행되었습니다. 일부 환경에서는 버전 호환성 확인이 필요할 수 있습니다.

    CPU: Intel(R) Core(TM) i9-13900KF
    GPU: Nvidia GeForce RTX 4090, CUDA 12.1
    OS: Ubuntu 24.04 LTS
    Conda: 25.5.1

## Installation

이 저장소에서 제공하는 모듈을 실행하기 위해 Conda 기반 환경을 구성합니다.

만약, Conda가 설치되어 있지 않다면 아래 링크에 접속하여 설치 후 단계를 진행합니다.

[🔗 아나콘다 다운로드](https://www.anaconda.com/download/success) 또는 [🔗 미니콘다 다운로드](https://www.anaconda.com/docs/getting-started/miniconda/main)

**Step 1**. Conda 가상환경 생성 및 활성화

```bash
conda create --name ngc-pipe-action python=3.8 -y
conda activate ngc-pipe-action
```

**Step 2**. 라이브러리 설치

```bash
pip install -r requirements.txt
pip install -v -e .
```

**Step 3**. MMLab 라이브러리 설치

```bash
pip install -U openmim
mim install mmengine==0.10.7
mim install mmcv==2.1.0
mim install mmdet==3.3.0
```

## Directory Structure

```
.
├── configs             : 모델 설정 관련 디렉토리
├── data
│   ├── input           : 원본 동영상 디렉토리
│   └── output          : 처리 결과 디렉토리
├── mmpose              : mmpose 모듈
├── model-index.yml     : 호환 모델 목록
├── requirements.txt    : 라이브러리 목록
├── run.py              : 메인 코드
├── setup.cfg           : 환경 설정 파일
├── setup.py            : 환경 설정 파일
└── requirements        : 라이브러리 목록 디렉토리
```

## Run

데이터셋 구축을 위해 아래 명령어를 실행합니다.

```bash
python run.py --pose2d rtmo
```

`run.py`는 아래 단계를 거쳐 데이터셋을 구축합니다.

**Step 1**. 인자 설정: 모델 초기화 인자와 호출 인자를 설정합니다.

**Step 2**. 동영상 로드: `data/input` 디렉토리에서 `.mp4`, `.avi`, `.mov` 파일들을 찾아 동영상 목록(파일 경로)를 생성합니다.

**Step 3**. 포즈 추론: 동영상 목록의 동영상에서 포즈를 추론합니다. 이 포즈는 COCO Keypoint 표기법을 따릅니다.

-   `output/pose`: 프레임별 BBOX/포즈로 구성된 `json`파일
-   `output/vis`: 원본 영상 위에 포즈 정보가 오버랩된 동영상 파일

**Step 4**. 데이터셋 변환: `Step 3`에서 생성된 포즈 데이터`data/output/pose`를 학습 데이터셋 형식으로 변환합니다.

-   `output/dataset.pkl`: 학습 데이터셋 형식으로 변환된 `pickle`파일

## Train Dataset Structure

```python
dataset = {
    'split': {
        'train': [], # 학습용 동영상 파일명
        'val': [], # 검증용 동영상 파일명
    },
    'annotations': [
        {
            'frame_dir': '', # 동영상 이름
            'label': '', # 레이블
            'img_shape': (0, 0), # 이미지 크기
            'original_shape': (0, 0), # 원본 크기
            'total_frames': 0, # 총 프레임 수
            'keypoint': [], # 키포인트
            'keypoint_score': [] # 키포인트 정확도 점수
        },
        ...
    ]
}
```
