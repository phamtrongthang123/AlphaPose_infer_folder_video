from ast import arg
from pathlib import Path 
import os
import argparse
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--inpdir', dest='inpdir',
                    help='input-directory', default="inputs/")
parser.add_argument('--outdir', dest='outdir',
                    help='output-directory', default="outputs/")
parser.add_argument('--gpu', type=str, dest='gpu', default="0",
                    help='choose which cuda device to use by index e.g. 0 (input -1 for cpu only)')
args = parser.parse_args()
os.makedirs(args.outdir,exist_ok=True)
datapath = Path(args.inpdir)
for vidf in datapath.glob("*"):
    vidp = list(vidf.glob("*"))[0] 
    os.makedirs(os.path.join(args.outdir,vidf.name), exist_ok=True)
    multiplier = 2
    os.system(f'python scripts/demo_inference.py --gpus {args.gpu} --posebatch {64*multiplier} --detbatch {5*multiplier} --cfg configs/halpe_coco_wholebody_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml --checkpoint pretrained_models/multi_domain_fast50_regression_256x192.pth --video="{str(vidp)}" --outdir "{os.path.join(args.outdir,vidf.name)}" --save_video --pose_track --sp')
    # break