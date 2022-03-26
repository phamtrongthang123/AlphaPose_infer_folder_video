from pathlib import Path 
import os

from torch import mul 

datapath = Path('/media/data/aioz-thang/main_dev/dataset/dataset')
for vidf in datapath.glob("*"):
    vidp = list(vidf.glob("*"))[0] 
    os.makedirs(f"./outputs/{vidf.name}", exist_ok=True)
    multiplier = 2
    os.system(f'python scripts/demo_inference.py --gpus 0 --posebatch {64*multiplier} --detbatch {5*multiplier} --cfg configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml --checkpoint pretrained_models/multi_domain_fast50_regression_256x192.pth --video="{str(vidp)}" --outdir "outputs/{vidf.name}" --save_video --pose_track --sp')
    # break