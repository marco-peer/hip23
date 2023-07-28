
import os, argparse
from glob import glob

import numpy as np
from PIL import Image

import torch, torchvision
from tqdm import tqdm

from models.convformer import ModelWrapper, Model
from utils.aug import Resize
from utils.utils import GPU


def get_test_tf():
    IMG_SIZE = (512,128)
    test_tf = torchvision.transforms.Compose([
        Resize(max(IMG_SIZE)),
        torchvision.transforms.CenterCrop(IMG_SIZE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        )
    ])
    return test_tf

def infer(args):
    print(f'Set GPU to {args.gpuid}')
    GPU.set(args.gpuid, 400)

    src = args.src_path
    mode = args.mode

    get_model = lambda x : os.path.join(src, f'models/{x}.pt')
    get_imgs = lambda x: os.path.join(src, f'{x}')

    # init Model
    model = Model(img_size=(512,128), out_rows=4)
    model = ModelWrapper(model, num_classes=5, img_size=(512,128)).cuda()

    # load model and images
    print(f'Loading model from {get_model(mode)} for {mode} images..')
    model.load_state_dict(torch.load(get_model(mode))['model_state_dict'])
    model.eval()

    imgs = sorted(list(glob(f'{get_imgs(mode)}/**/*.jpg', recursive=True)))
    print(f'Found {len(imgs)} images in {get_imgs(mode)}')

    tf = get_test_tf()
    embs = []
    with torch.no_grad():
        for img in tqdm(imgs):
            i = Image.open(img).convert('RGB')
            emb = model(tf(i).unsqueeze(0).cuda())
            embs.append(emb.detach().cpu().numpy())
        embs = np.concatenate(embs)

    return embs

if __name__ == '__main__':
    ### Inference script to extract embeddings from images
    ### supports mode {color, sauvola, unet}
    ### src_path is expected to contain the following directories:
    ####### models - contains the {mode}.pt file
    ####### image directory - contains the images to be embedded, named after the mode

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='color')
    parser.add_argument('--gpuid', default='3', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--src_path', default='/data/mpeer/papyri_1200/hip23-data/', type=str)

    args = parser.parse_args()
    embeddings = infer(args)
    print(f'Extracted {embeddings.shape[0]} embeddings for {args.mode} images')
