import os
from skimage.io import imread, imshow, show
from typing import List

path = './tmp/Food_101_Dataset/food-101/'


def get_files_recursive(filepath: str) -> List[str]:
    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(filepath)):
        if i%250 == 0:
            print(i, 'folders sprocessed')
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
    print(f'Found {len(listOfFiles)} files')
    cleaned = [file for file in listOfFiles if file.endswith('.jpg')]
    print(f'{len(cleaned)} of whitch are actually jpg-files ({round(len(cleaned) / len(listOfFiles), 5)*100}%)')
    return cleaned


cleaned = get_files_recursive(path)
ressult = []
for i, img in enumerate(cleaned):
    if i%1000 == 0:
        print(100*round(i / len(cleaned), 5), '%    processed', i, 'of', len(cleaned), 'Images')
    try:
        im = imread(img, as_gray=True)
        #imshow(im)
        #show()
        #break
        ressult.append(list(im.shape))
    except:
        print('Error reading ', img)

import numpy as np
print('Average Resolution is', np.mean(ressult, axis=0))
