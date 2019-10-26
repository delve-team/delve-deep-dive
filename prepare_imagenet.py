from os import rename, listdir, mkdir
from os.path import exists, join, dirname
from shutil import rmtree, copyfile

ROOT = "G:\\ImageNet\\"
mapping_file = join(ROOT, 'mapping.txt')
coded_labels_file = join(ROOT, 'coded_labels_val.txt')
val_file_names = listdir(join(ROOT, 'val'))
val_file_path = [join(ROOT, 'val', file) for file in val_file_names]
val_file_path.sort()
val_file_names.sort()

if not exists(join(ROOT, 'valid')):
    mkdir(join(ROOT, 'valid'))

valid_target = join(ROOT, 'valid')


def get_mapping(mapping_file: str):
    id_to_wordnet = {}
    wordnet_to_name = {}
    with open(mapping_file, 'r') as fp:
        for line in fp:
            splitted = line.split(' ')
            id_to_wordnet[int(splitted[1])] = splitted[0]
            wordnet_to_name[splitted[0]] = splitted[2].replace('\n', '')
            print(splitted[1], splitted[0])
            print(splitted[0], splitted[2].replace('\n', ''))
    return id_to_wordnet, wordnet_to_name

def get_classes_raw(coded_labels_file):
    with open(coded_labels_file, 'r') as fp:
        result = []
        for line in fp:
            result.append(int(line))
    return result

id_to_wordnet, wordnet_to_name = get_mapping(mapping_file)
print('Lengths:', len(id_to_wordnet), len(wordnet_to_name))
raw_classes = get_classes_raw(coded_labels_file)

for i, (file_name, data_point, label) in enumerate(zip(val_file_names, val_file_path, raw_classes)):
    directory_name = id_to_wordnet[label]
    directory_path = join(valid_target, directory_name)
    if not exists(directory_path):
        mkdir(directory_path)
    new_file_path = join(directory_path, file_name)
    copyfile(data_point, new_file_path)
    print(i, file_name, label)
