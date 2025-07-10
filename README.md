# Pipeline Action

이 저장소는 행동 추정 모델을 학습시키기 위한 데이터셋을 구축하는 모듈을 제공합니다.

### Note

    모든 테스트는 다음 환경에서 진행되었습니다. 일부 환경에서는 버전 호환성 확인이 필요할 수 있습니다.

    CPU: Intel(R) Core(TM) i9-13900KF
    GPU: Nvidia GeForce RTX 4090, CUDA 12.1
    OS: Ubuntu 24.04 LTS
    Conda: 24.9.2

## Installation

이 저장소에서 제공하는 모듈을 실행하기 위해 Conda 기반 환경을 구성합니다.

만약, Conda가 설치되어 있지 않다면 아래 링크에 접속하여 설치 후 단계를 진행합니다.

[🔗 아나콘다 다운로드](https://www.anaconda.com/download/success)

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

## Demo

```bash
python run.py data/demo/demo.mp4 --pose2d rtmo
```
