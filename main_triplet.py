
import argparse, os, logging, math, random
import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn
from torch import optim

from tqdm import tqdm

from pytorch_metric_learning import samplers, miners, losses
from triplet_loss import TripletLoss

from utils.utils import GPU, seed_everything, getLogger, save_model, cosine_scheduler, load_config, load_yaml
from utils.aug import Resize, RandomApply
from utils.dataset import Subset
from utils.utils_train import print_info, model_zoo

from dataloading.writer_zoo import WriterZoo
from evaluators.retrieval import Retrieval

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def get_train_tf(args):
    IMG_SIZE = args['img_size']
    if type(IMG_SIZE) != list:
        IMG_SIZE = [IMG_SIZE, IMG_SIZE]

    train_tf = torchvision.transforms.Compose([
        Resize(max(IMG_SIZE)),
        torchvision.transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.1)),
        RandomApply(
          torchvision.transforms.GaussianBlur((3, 3), (1.0, 2.0)),
          p = 0.5
        ),
        torchvision.transforms.RandomAffine(5, translate=(0.1,0.1)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225]))
    ])
    return train_tf

def get_test_tf(args):
    IMG_SIZE = args.get('test_img_size', args['img_size'])
    if type(IMG_SIZE) != list:
        IMG_SIZE = [IMG_SIZE, IMG_SIZE]

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


def inference(model, ds, args):
    model.eval()
    loader = torch.utils.data.DataLoader(ds, num_workers=8, pin_memory=True, batch_size=args['test_batch_size'])

    feats = []
    pages = []
    writers = []

    for sample, labels in tqdm(loader, desc='Inference'):
        w, p, f = labels

        writers.append(w)
        pages.append(p)
        sample = sample.cuda()

        with torch.no_grad():
            emb = model(sample)
            emb = torch.nn.functional.normalize(emb)
        feats.append(emb.detach().cpu().numpy())
    
    feats = np.concatenate(feats)
    writers = np.concatenate(writers)
    pages = np.concatenate(pages)   

    return feats, writers, pages

def validate(model, val_ds, args, eval_label='writer'):
    desc, writer, pages = inference(model, val_ds, args)
    print('Inference done')

    if args['pca_dim'] != -1 and desc.shape[0] > 1e3:
        pca_dim = min(args['pca_dim'], desc.shape[0])
        pca = PCA(pca_dim, whiten=True)
        print(f"Fitting pca with dim {pca_dim}.")

        if np.isnan(desc).any() or (not np.isfinite(desc).all()):
            logging.info("Detected NaNs or Inf!")
        desc = np.nan_to_num(desc)

        try:
            pfs_tf = pca.fit_transform(desc)
            pfs_tf = normalize(pfs_tf, axis=1)
        except:
            logging.info("Found nans in input. skipping pca")
            pfs_tf = normalize(pfs_tf, axis=1)
    else:
        pfs_tf = normalize(desc, axis=1)

    if eval_label == 'image':
        logging.info('Evaluating on image level')
        wp = np.stack([writer,pages],axis=1)
        wps = [list(i[0]) for i in list(zip(wp))]
        labels = np.array([wps.index(i) for i in wps])
    else:
        logging.info('Evaluating on writer level')
        labels = np.array(writer)
    logging.info(f'Found {len(list(set(labels)))} different instances.')


    _eval = Retrieval()
    res, _ = _eval.eval(pfs_tf, labels)
    meanavp = res['map']
    top1 = res['top1']
    return meanavp, top1


def train_one_epoch(model, train_ds, triplet_loss, optimizer, scheduler, epoch, args, logger, scaler):

    model.train()
    model = model.cuda()

    # set up the triplet stuff
    sampler = samplers.MPerClassSampler(np.array(train_ds.dataset.labels[args['train_label']])[train_ds.indices], args['train_options']['sampler_m'], length_before_new_iter=args['train_options']['length_before_new_iter']) #len(ds))
    train_triplet_loader = torch.utils.data.DataLoader(train_ds, sampler=sampler, pin_memory=True, batch_size=args['train_options']['batch_size'], drop_last=True, num_workers=16)
    # miner = miners.MultiSimilarityMiner(epsilon=0.1) if args['train_options']['loss'] == 'msloss' else None
    # miner = miners.TripletMarginMiner(margin=args['train_options']['margin'], type_of_triplets=args['train_options']['type_of_triplets'])
    # miner = miners.BatchEasyHardMiner()
    miner = None
    pbar = tqdm(train_triplet_loader)
    pbar.set_description('Epoch {} Training'.format(epoch))
    iters = len(train_triplet_loader)
    logger.log_value('Epoch', epoch, commit=False)

    tile = lambda x: torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(x.cpu(), nrow=12))

    for i, (samples, label) in enumerate(pbar):
        it = iters * epoch + i
        for i, param_group in enumerate(optimizer.param_groups):
            if it > (len(scheduler) - 1):
                param_group['lr'] = scheduler[-1]
            else:
                param_group["lr"] = scheduler[it]
            
   
        samples = samples.cuda()
        samples.requires_grad=True

        l = label[0]
        l = l.cuda()

        if scaler:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                emb = model(samples)
                if miner:
                    triplets = miner(emb, l)
                    loss = triplet_loss(emb, l, triplets)
                else:
                    loss = triplet_loss(emb, l, emb, l)
        else:
            emb = model(samples)
            if miner:
                triplets = miner(emb, l)
                loss = triplet_loss(emb, l, triplets)
            else:
                loss = triplet_loss(emb, l, emb, l)

        logger.log_value(f'loss', loss.item())
        logger.log_value(f'lr', optimizer.param_groups[0]['lr'])

        # mixed precision stuff
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            optimizer.zero_grad()
        # if epoch < 2:
        #     tile(samples[:12]).save(os.path.join(logger.log_dir, 'samples.png'))


    torch.cuda.empty_cache()
    return model

def train(model, train_ds, val_ds, args, logger, optimizer):

    epochs = args['train_options']['epochs']

    niter_per_ep = math.ceil(args['train_options']['length_before_new_iter'] / args['train_options']['batch_size'])
    lr_schedule = cosine_scheduler(args['optimizer_options']['base_lr'], args['optimizer_options']['final_lr'], epochs, niter_per_ep, warmup_epochs=args['optimizer_options']['warmup_epochs'], start_warmup_value=0)
    
    best_epoch = -1
    best_map, best_top1 = validate(model, val_ds, args)

    print(f'Val-mAP: {best_map}')
    logger.log_value('Val-mAP', best_map)
    logger.log_value('Val-Top1', best_top1)

    if args['train_options']['loss'] == 'triplet':
        loss = TripletLoss(margin=args['train_options']['margin'])
        # loss = losses.TripletMarginLoss(margin=args['train_options']['margin'],swap=True)
        print('Using Triplet Loss')
    elif args['train_options']['loss'] == 'msloss':
        loss = losses.MultiSimilarityLoss()
        print('Using MSLoss')
    else:
        raise ValueError("Unknown Loss")
    
    scaler = torch.cuda.amp.GradScaler() if args['train_options'].get('mixed_precision', False) else None
    if scaler:
        logging.info('Using mixed precision training')

    for epoch in range(epochs):
        model = train_one_epoch(model, train_ds, loss, optimizer, lr_schedule, epoch, args, logger, scaler)
        mAP, top1 = validate(model, val_ds, args)

        logger.log_value('Val-mAP', mAP)
        logger.log_value('Val-Top1', top1)

        print(f'Val-mAP: {mAP}')
        print(f'Val-Top1: {top1}')


        if mAP > best_map:
            best_epoch = epoch
            best_map = mAP
            save_model(model, optimizer, epoch, os.path.join(logger.log_dir, 'model.pt'))


        if (epoch - best_epoch) > args['train_options']['callback_patience']:
            break

    # load best model
    checkpoint = torch.load(os.path.join(logger.log_dir, 'model.pt'))
    print(f'''Loading model from Epoch {checkpoint['epoch']}''')
    model.load_state_dict(checkpoint['model_state_dict'])    
    model.eval() 
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def prepare_logging(args):
    os.path.join(args['log_dir'], args['super_fancy_new_name'])
    Logger = getLogger(args["logger"])
    logger = Logger(os.path.join(args['log_dir'], args['super_fancy_new_name']), args=args)
    logger.log_options(args)
    return logger

def train_val_split(dataset, args, prop = 0.9):
    authors = list(set(dataset.labels['writer']))
    random.shuffle(authors)

    train_len = math.floor(len(authors) * prop)
    train_authors = authors[:train_len]
    val_authors = authors[train_len:]

    print(f'{len(train_authors)} authors for training - {len(val_authors)} authors for validation')

    train_idxs = []
    val_idxs = []

    for i in tqdm(range(len(dataset)), desc='Splitting dataset'):
        w = dataset.get_label(i)[0]
        if w in train_authors:
            train_idxs.append(i)
        if w in val_authors:
            val_idxs.append(i)

    train = Subset(dataset, train_idxs, transform=get_train_tf(args))
    val = Subset(dataset, val_idxs, transform=get_test_tf(args))

    return train, val

def get_optimizer(args, model):
    if args['optimizer_options']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args['optimizer_options']['base_lr'],
                        weight_decay=args['optimizer_options']['wd'])
    if args['optimizer_options']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args['optimizer_options']['base_lr'],
                            weight_decay=args['optimizer_options']['wd'])
    return optimizer


def main(args):
    logger = prepare_logging(args)
    logger.update_config(args)

    model = model_zoo(args)
    if torch.__version__ == '2.0.0' and args['compile_model']:
        logging.info("Pytorch 2.0 detected -> compiling model")
        model = torch.compile(model)
    model.train()
    model = model.cuda()

    optimizer = get_optimizer(args, model)

    if args['checkpoint']:
        print(f'''Loading model from {args['checkpoint']}''')
        checkpoint = torch.load(args['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'])    
        model.eval() 

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    

    if not args['only_test']:
        train_dataset = None
        if args['trainset']:
            train_dataset = WriterZoo.get(**args['trainset'])
        
        train_ds, val_ds = train_val_split(train_dataset, args)
        model, optimizer = train(model, train_ds, val_ds, args, logger, optimizer)

    # testing
    test_ds = WriterZoo.get(**args['testset']).TransformImages(transform=get_test_tf(args))

    test_map , test_top1 = validate(model, test_ds, args, eval_label=args.get('eval_label', 'writer'))

    print(f'Test-mAP: {test_map}')
    print(f'Test-Top1: {test_top1}')

    logger.log_value("Test-MAP", test_map)
    logger.log_value("Test-Top1", test_top1)
    logger.finish()


def main_cross_validate(args):
    logger = prepare_logging(args)
    logger.update_config(args)

    dataset = WriterZoo.get(**args['dataset'])
    cross_val = load_yaml(args['kfold_config'])[args['kfold']]
    splits = cross_validation_splits(dataset, list(cross_val.values()))

    def flatten(l):
        return [item for sublist in l for item in sublist]

    maps, top1 = [], []
    for idx in range(len(splits)):
        train_indices = splits[idx]
        val_indices = train_indices[0::int(1/args['val_percentage'])]
        train_indices = [t_idx for t_idx in train_indices if t_idx not in val_indices]
        test_indices = flatten(splits[:idx] + splits[idx+1:])
        

        train_ds = Subset(dataset, train_indices, get_train_tf(args))
        val_ds = Subset(dataset, val_indices, get_test_tf(args))
        test_ds = Subset(dataset, test_indices, get_test_tf(args))

        print_info(train_ds, val_ds, test_ds, idx+1, len(splits))
        
        model = model_zoo(args)
        if torch.__version__ == '2.0.0' and args['compile_model']:
            logging.info("Pytorch 2.0 detected -> compiling model")
            model = torch.compile(model)
        model.train()
        model = model.cuda()

        optimizer = get_optimizer(args, model)

        if args['checkpoint']:
            print(f'''Loading model from {args['checkpoint']}''')
            checkpoint = torch.load(args['checkpoint'])
            model.load_state_dict(checkpoint['model_state_dict'])    
            model.eval() 

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    

        model, optimizer = train(model, train_ds, val_ds, args, logger, optimizer)

        # testing
        test_map , test_top1 = validate(model, test_ds, args, eval_label=args.get('eval_label', 'writer'))
        maps.append(test_map)
        top1.append(test_top1)

        print(f'Test-mAP k={idx}: {test_map}')
        print(f'Test-Top1 k={idx}: {test_top1}')

        logger.log_value(f"Test-MAP-{idx}", test_map)
        logger.log_value(f"Test-Top1-{idx}", test_top1)

    logger.log_value(f"Test-MAP", np.mean(maps))
    logger.log_value(f"Test-Top1", np.mean(top1))
    logger.finish()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config/config.yml')
    parser.add_argument('--only_test', default=False, action='store_true',
                        help='only test')
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpuid', default='1', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', default=2174, type=int,
                        help='seed')

    args = parser.parse_args()

    if type(args.config) != list:
        args.config = [args.config]
        
    config = load_config(args)[0]

    GPU.set(args.gpuid, 400)
    cudnn.benchmark = True
    
    seed_everything(args.seed)
    if config.get('kfold', None):
        logging.info('Running kfold cross validation')
        main_cross_validate(config)
    else:
        main(config)