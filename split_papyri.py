import argparse
import os, glob, shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str)
parser.add_argument('--outdir', type=str)

args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

imgs = list(glob.glob(f"{os.path.join(args.indir, '**/*.jpg')}", recursive=True))
print(f'found {len(imgs)} images.')

splits = [
    ['daueit', 'anouphis', 'pilatos', 'ieremias', 'victor'],
    ['kyros', 'psates', 'philotheos', 'menas', 'amais'],
    ['kollouthos', 'andreas', 'konstantinos', 'dioscorus', 'aparhasios'],
    ['hermauos', 'abraamios', 'dios', 'theodosios', 'isak']
]

for img in imgs:
    imgname = Path(img).name.split("_")[0]
    author = ''.join([i for i in imgname if not i.isdigit()]).lower()
    print(author)
    sx = [idx for idx, s in enumerate(splits) if author in s][0]

    out = os.path.join(args.outdir, f'set{sx}')
    if not os.path.exists(out):
        os.mkdir(out)
    shutil.copyfile(img, os.path.join(out, Path(img).name))