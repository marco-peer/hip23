import os, csv, argparse

from glob import glob

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.metrics import pairwise_distances

def stack_images_with_captions(images, captions, fig_width=20, save_path=None, image_name=None):
    """
    Stack images with their own captions and display them using Matplotlib.

    Parameters:
        images (list of ndarray): A list of NumPy arrays representing the images.
        captions (list of str): A list of strings representing the captions for each image.
        num_cols (int): Number of columns in the subplot grid (default is 3).
        fig_width (int): Width of the figure in inches (default is 15).

    Returns:
        None
    """
    num_images = len(images)
    num_cols = num_images
    fig, axarr = plt.subplots(1, num_cols, figsize=(fig_width, fig_width/num_cols))

    for i, (image, caption) in enumerate(zip(images, captions)):
        ax = axarr[i]
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.set_title(caption)

        if i == 0:
            border_color = 'blue'
        elif captions[0].split("_")[0] == captions[i].split("_")[0]:
            border_color = 'green' 
        else:
            border_color = 'red'

        border_width = 5  # You can adjust the border width as needed
        rect = patches.Rectangle((0, 0), image.shape[1], image.shape[0], linewidth=border_width, edgecolor=border_color, facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, image_name))
    else:
        plt.show()

def retrieve_image(query, dists, labels, imgs, args):
    labels = np.array(labels, dtype='object')
    nns = np.argsort(dists, axis=1)
    idx = np.where(labels == query)[0]
    if not idx:
        raise ValueError(f'Could not find {query} in labels')
    
    query_nns = nns[idx, 1:6]
    query_nn_names = labels[query_nns][0]
    print(f'{query} : {query_nn_names}')

    query_image = np.array(Image.open(imgs[idx[0]]))
    nn_images = [np.array(Image.open(imgs[idi])) for idi in query_nns[0]]

    images = [query_image] + nn_images
    captions = [query]
    captions.extend(query_nn_names)
    stack_images_with_captions(images, captions, save_path=args.save_path, image_name = f'{query}_{args.mode}.png')

def calculate_distances(args):
    src, mode = args.src_path, args.mode
    get_embs = lambda x: os.path.join(src, f'embeddings/{x}.npy')
    get_imgs = lambda x : f'/data/mpeer/papyri_1200/{mode}'
    label_csv = os.path.join(src, f'embeddings/labels.csv')

    with open(label_csv, newline='') as csvfile:
        labels = list(csv.reader(csvfile))[0]
    embs = np.load(get_embs(mode))
    imgs = sorted(list(glob(f'{get_imgs(mode)}/*/**.jpg', recursive=True)))
    print(f'found {len(imgs)} images')

    dists = pairwise_distances(embs, metric='cosine')

    return dists, labels, imgs

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', default='/data/mpeer/papyri_1200/hip23-data/', type=str)
    parser.add_argument('--mode', type=str, default='color')
    parser.add_argument('--query', default='Dioscorus_4_13', type=str,
                        help='fragment name as image')
    parser.add_argument('--save_path', default='retrieval_images', type=str)

    args = parser.parse_args()

    dists, labels, imgs = calculate_distances(args)
    retrieve_image(args.query, dists, labels, imgs, args)