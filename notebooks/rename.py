import glob, os
from pathlib import Path
all_imgs = sorted(list(glob.glob('/data/mpeer/papyri_1200/binarized_unet_2/**/*.jpg', recursive=True)))

        
s1 = ['daueit', 'anouphis', 'pilatos', 'ieremias', 'victor']
s2 = ['kyros', 'psates', 'philotheos', 'menas', 'amais']
s3 = ['kollouthos', 'andreas', 'konstantinos', 'dioscorus', 'aparhasios']
s4 = ['hermauos', 'abraamios', 'dios_', 'theodosios', 'isak']

s1,s2,s3,s4 = [['daueit', 'abraamios', 'philotheos', 'konstantinos', 'victor'], ['pilatos', 'isak', 'ieremias', 'hermauos', 'andreas'], ['aparhasios', 'dios_', 'psates', 'menas', 'anouphis'], ['amais', 'theodosios', 'kollouthos', 'kyros', 'dioscorus']]

# s1,s2,s3,s4,s5 = [['aparhasios', 'philotheos', 'kyros', 'kollouthos'], ['hermauos', 'isak', 'psates', 'konstantinos'], ['daueit', 'menas', 'andreas', 'abraamios'], ['ieremias', 'dios', 'amais', 'anouphis'], ['pilatos', 'dioscorus', 'theodosios', 'victor']]

nums = [0,0,0,0]
for img in list(set(all_imgs)):
    author_img = Path(img).name.lower()
    for i, s in enumerate([s1,s2,s3,s4]):
        for author in s:
            if author_img.startswith(author):
                nums[i] += 1
                break

print(nums, sum(nums))
print(sorted(s1))
print(sorted(s2))
print(sorted(s3))
print(sorted(s4))