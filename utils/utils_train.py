import logging
from pathlib import Path
import torch

def cross_validation_splits(dataset, splits):
    split_imgs = [[] for i in range(len(splits))]
    for img_idx, img in enumerate(dataset.imgs):
        imgname = Path(img).name.split("_")[0]
        author = ''.join([i for i in imgname if not i.isdigit()]).lower()
        sx = [idx for idx, s in enumerate(splits) if author in s][0]

        split_imgs[sx].append(img_idx)
    
    for split in split_imgs:
        split.sort()

    return split_imgs

def print_info(train, val, test, split_id, total_splits):
    logging.info(10*"-")
    logging.info(f'K-Fold Crossvalidation Run {split_id+1}/{total_splits}')
    logging.info(10*"-")

    logging.info(f'Using {len(train)} files for training')
    logging.info(f'Using {len(val)} files for validation')
    logging.info(f'Using {len(test)} files for testing')
    logging.info(10*"-")

def model_zoo(args):
    name, img_size, model_config = args['model']['name'], args['img_size'], args['model']

    if name == 'resnet34':
        from models.convformer import ResNet34
        model = ResNet34()

    if name == 'mixconv':
        import models.convformer
        model = models.convformer.Model(img_size=img_size,mix_depth=args['model']['mix_depth'], out_channels=args['model']['out_channels'], out_rows=args['model']['out_rows'])
        
        if args['model']['drop_mixing']:
            model.agg.mix = torch.nn.Identity()
            logging.info("Removed mixer!")

    if name == 'resnet34mixer':
        from models.convformer import ResNet34Mixer
        model = ResNet34Mixer(img_size=img_size)
        
    print(f'Initiated Model {name}')
    return model