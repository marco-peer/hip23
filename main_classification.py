
import argparse, os, logging, math, random
import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn
from torch import optim

from tqdm import tqdm

from utils.utils import GPU, seed_everything, getLogger, save_model, cosine_scheduler, load_config, load_yaml
from utils.aug import Resize
from utils.dataset import Subset
from utils.utils_train import print_info, cross_validation_splits, model_zoo

from dataloading.writer_zoo import WriterZoo
from evaluators.retrieval import Retrieval

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from models.convformer import ModelWrapper

def get_train_tf(args):
    IMG_SIZE = args['img_size']
    if type(IMG_SIZE) != list:
        IMG_SIZE = [IMG_SIZE, IMG_SIZE]

    train_tf = torchvision.transforms.Compose([
        Resize(max(IMG_SIZE)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop(IMG_SIZE),
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

def validate_classification(model, val_ds, args):
    model.eval()
    accuracy_score = 0
    loader = torch.utils.data.DataLoader(val_ds, num_workers=8, pin_memory=True, batch_size=args['test_batch_size'])
    for sample, labels in tqdm(loader, desc='Inference'):
        sample = sample.cuda()

        w, p, f = labels
        score = model(sample)
        y_ = torch.argmax(score, dim=1).cpu()
        accuracy_score += torch.sum(y_ == w).item()

    accuracy_score = accuracy_score / len(val_ds)
    print(f'Accuracy: {accuracy_score}') 
    return accuracy_score

def validate_retrieval(model, val_ds, args, eval_label='writer'):
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


def train_one_epoch(model, train_ds, classification_loss, optimizer, scheduler, epoch, args, logger, scaler):

    model.train()
    model = model.cuda()

    train_loader = torch.utils.data.DataLoader(train_ds, pin_memory=True, batch_size=args['train_options']['batch_size'], drop_last=True, num_workers=16, shuffle=True)

    pbar = tqdm(train_loader)
    pbar.set_description('Epoch {} Training'.format(epoch))
    iters = len(train_loader)
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
        if getattr(model, 'lookup', None):
            l = torch.tensor([model.lookup[li.item()] for li in l])

        l = l.cuda()
        target = torch.nn.functional.one_hot(l, num_classes = model.classifier.out_features).float()

        if scaler:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                scores = model(samples)
        else:
            scores = model(samples)

        loss = classification_loss(target, scores)
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

        if epoch == 0 and i == 1:
            tile(samples[:12]).save(os.path.join(logger.log_dir, 'samples.png'))

    torch.cuda.empty_cache()
    return model

def train(model, train_ds, val_ds, args, logger, optimizer):

    epochs = args['train_options']['epochs']

    niter_per_ep = math.ceil(len(train_ds) / args['train_options']['batch_size'])
    lr_schedule = cosine_scheduler(args['optimizer_options']['base_lr'], args['optimizer_options']['final_lr'], epochs, niter_per_ep, warmup_epochs=args['optimizer_options']['warmup_epochs'], start_warmup_value=0)
    
    best_epoch = -1
    validate = validate_retrieval if args['eval_mode'] == 'retrieval' else validate_classification

    if args['eval_mode'] == 'retrieval':
        best_performance, best_top1 = validate(model, val_ds, args)
        print(f'Val-mAP: {best_performance}')
        print(f'Val-Top1: {best_top1}')

        logger.log_value('Val-mAP', best_performance)
        logger.log_value('Val-Top1', best_top1)
    else:
        best_performance = validate(model, val_ds, args)
        print(f'Accuracy: {best_performance}')
        logger.log_value('Val-Acc', best_performance)

    loss = torch.nn.CrossEntropyLoss()      
    
    scaler = torch.cuda.amp.GradScaler() if args['train_options'].get('mixed_precision', False) else None
    if scaler:
        logging.info('Using mixed precision training')

    for epoch in range(epochs):
        model = train_one_epoch(model, train_ds, loss, optimizer, lr_schedule, epoch, args, logger, scaler)

        if args['eval_mode'] == 'retrieval':
            performance, top1 = validate(model, val_ds, args)
            logger.log_value('Val-mAP', performance)
            logger.log_value('Val-Top1', top1)
            print(f'Val-mAP: {performance}')
            print(f'Val-Top1: {top1}')
        else:
            performance = validate(model, val_ds, args)
            logger.log_value('Val-Acc', performance)

        if performance > best_performance:
            best_epoch = epoch
            best_performance = performance
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

def train_val_test_split(dataset, args, prop = [0.2, 0.1]):
    def flatten(l):
        return [item for sublist in l for item in sublist]
    authors = list(set(dataset.labels['writer']))
    print(f'{len(authors)} authors included in the dataset')
    idx_per_author = {}
    for i in tqdm(range(len(dataset)), desc='Splitting dataset'):
        w = dataset.get_label(i)[0]
        p = dataset.get_label(i)[1]

        if not idx_per_author.get(w, None):
            idx_per_author[w] = {}
        if not idx_per_author[w].get(p, None):
            idx_per_author[w][p] = []
        idx_per_author[w][p].append(i)
    
    train_indices, val_indices, test_indices = [], [], []
    
    for _, page_ids in idx_per_author.items():
        indices = list(page_ids.keys())
        random.shuffle(indices)
        if len(indices) == 1:
            train_indices.extend(page_ids[indices[0]])
            continue

        if len(indices) == 2:
            train_indices.extend(page_ids[indices[0]])
            test_indices.extend(page_ids[indices[1]])
            continue

        train_idxs = max(int(len(page_ids) * prop[0]), 1)
        val_idxs = int(len(page_ids) * prop[1])
        test_idxs = int(len(page_ids) * (1-sum(prop)))

        train_indices.extend(flatten([page_ids[idx] for idx in indices[:train_idxs]]))
        if val_idxs:
            val_indices.extend(flatten([page_ids[idx] for idx in indices[train_idxs:(train_idxs+val_idxs)]]))
            
        test_indices.extend(flatten([page_ids[idx] for idx in indices[train_idxs+val_idxs:]]))


    print(f'{len(train_indices)} files for training')
    print(f'{len(val_indices)} files for validation')
    print(f'{len(test_indices)} files for testing')

    train = Subset(dataset, train_indices, transform=get_train_tf(args))
    val = Subset(dataset, val_indices, transform=get_test_tf(args))
    test = Subset(dataset, test_indices, transform=get_test_tf(args))

    return train, val, test

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

    test_accs = []
    for seed in [42, 2174, 1000, 7200, 8]:
        seed_everything(seed)
        logging.info(f"Run with seed {seed}")

        model = model_zoo(args)
        if torch.__version__ == '2.0.0' and args['compile_model']:
            logging.info("Pytorch 2.0 detected -> compiling model")
            model = torch.compile(model)

        model.train()

        if not args['only_test']:
            train_dataset = None
            if args['dataset']:
                train_dataset = WriterZoo.get(**args['dataset'])
                train_ds, val_ds, test_ds = train_val_test_split(train_dataset, args, prop=args['dataset_split'])
        
        class_idxs = list(set(np.array(train_ds.dataset.labels['writer'])[np.array(train_ds.indices)]))
        lookup = {class_idx : i for i, class_idx in enumerate(class_idxs)}
        logging.info(f'{len(class_idxs)} authors for training')
        model = ModelWrapper(model, len(class_idxs), args['img_size'], mode=args['eval_mode'], lookup=lookup).cuda()
        model.train()

        optimizer = get_optimizer(args, model)


        if args['checkpoint']:
            print(f'''Loading model from {args['checkpoint']}''')
            checkpoint = torch.load(args['checkpoint'])
            model.load_state_dict(checkpoint['model_state_dict'])    
            model.eval() 

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    

        if not args['only_test']:
            model, optimizer = train(model, train_ds, val_ds, args, logger, optimizer)

        # testing
        if args['eval_mode'] == 'retrieval':
            test_map , test_top1 = validate_retrieval(model, test_ds, args)
            print(f'Test-mAP: {test_map}')
            print(f'Test-Top1: {test_top1}')

            logger.log_value("Test-MAP", test_map)
            logger.log_value("Test-Top1", test_top1)
        else:
            test_acc = validate_classification(model, test_ds, args)
            test_accs.append(test_acc)
            print(f'Test-Accuracy: {test_acc}')
            logger.log_value("Test-Accuracy", test_acc)

    logger.log_value("Mean-Test-Accuracy", np.mean(test_accs))
    logger.finish()



def main_cross_validate(args):
    logger = prepare_logging(args)
    logger.update_config(args)

    dataset = WriterZoo.get(**args['dataset'])
    cross_val = load_yaml(args['kfold_config'])[args['kfold']]
    splits = cross_validation_splits(dataset, list(cross_val.values()))

    def flatten(l):
        return [item for sublist in l for item in sublist]

    print(f"Authors: {list(dataset.label_to_int['writer'].keys())}")
    maps_writer, top1_writer = [], []
    maps_page, top1_page = [], []

    for idx in range(len(splits)):
        train_indices = splits[idx]
        val_indices = []
        writer, pages = [], []
        for train_idx in train_indices:
            writer.append(dataset.get_label(train_idx)[0])
            pages.append(dataset.get_label(train_idx)[1])
            

        pairs = list(zip(writer,pages))
        random.shuffle(pairs)
        pairs = np.array(pairs)
        val_pages = []
        for w in list(set(writer)):
            idx_w = np.where(pairs[:, 0] == w)[0]
            w_pages = np.unique(pairs[idx_w][: , 1])
            random.shuffle(w_pages)

            if len(w_pages) == 1:
                print(f'Only one page for {w}')
                continue
            else:
                rel = w_pages[:max(int(len(w_pages) * 0.2), 1)]
                for wi in rel:
                    val_pages.append([w, wi])

        for train_idx in train_indices:
            if [dataset.get_label(train_idx)[0], dataset.get_label(train_idx)[1]] in val_pages:
                val_indices.append(train_idx)

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

        class_idxs = list(set(np.array(train_ds.dataset.labels['writer'])[np.array(train_ds.indices)]))
        lookup = {class_idx : i for i, class_idx in enumerate(class_idxs)}
        logging.info(f'{len(class_idxs)} authors for training')
        model = ModelWrapper(model, len(class_idxs), args['img_size'], mode=args['eval_mode'], lookup=lookup).cuda()
        model.train()

        optimizer = get_optimizer(args, model)

        if args['checkpoint']:
            print(f'''Loading model from {args['checkpoint']}''')
            checkpoint = torch.load(args['checkpoint'])
            model.load_state_dict(checkpoint['model_state_dict'])    
            model.eval() 

            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])    

        model, optimizer = train(model, train_ds, val_ds, args, logger, optimizer)

        # testing
        test_writer_map , test_writer_top1 = validate_retrieval(model, test_ds, args)
        maps_writer.append(test_writer_map)
        top1_writer.append(test_writer_top1)

        page_map , page_top1 = validate_retrieval(model, test_ds, args, eval_label='image')
        maps_page.append(page_map)
        top1_page.append(page_top1)


        print(f'Test-Writer-mAP k={idx}: {test_writer_map}')
        print(f'Test-Writer-Top1 k={idx}: {test_writer_top1}')

        logger.log_value(f"Test-MAP-{idx}", test_writer_map)
        logger.log_value(f"Test-Top1-{idx}", test_writer_top1)

        print(f'Test-Page-mAP k={idx}: {page_map}')
        print(f'Test-Page-Top1 k={idx}: {page_top1}')

        logger.log_value(f"Test-Page-MAP-{idx}", page_map)
        logger.log_value(f"Test-Page-Top1-{idx}", page_top1)

    logger.log_value(f"Test-MAP", np.mean(maps_writer))
    logger.log_value(f"Test-Top1", np.mean(top1_writer))
    
    logger.log_value(f"Test-Page-MAP", np.mean(maps_page))
    logger.log_value(f"Test-Page-Top1", np.mean(top1_page))
    logger.finish()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='config/config_papyrow_classification.yml')
    parser.add_argument('--only_test', default=False, action='store_true',
                        help='only test')
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--gpuid', default='-1', type=str,
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
    if config.get('kfold', None) and config.get('eval_mode', None) == 'retrieval':
        logging.info('Running kfold cross validation')
        main_cross_validate(config)
    else:
        main(config)