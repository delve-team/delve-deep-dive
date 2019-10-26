from os.path import exists, join, dirname, curdir
from os import listdir, mkdir, sep
from shutil import rmtree, copyfile

ROOT = join(curdir, "tmp", "tiny-imagenet-200")
TGT_ROOT_VAL = join(ROOT, "val", "ds")
TGT_ROOT_TEST = join(ROOT, "test", "ds")
SRC_ROOT_VAL = join(ROOT, "val", "images")
SRC_ROOT_TEST = join(ROOT, "test", "images")

if exists(TGT_ROOT_VAL):
    rmtree(TGT_ROOT_VAL)
if exists(TGT_ROOT_TEST):
    rmtree(TGT_ROOT_TEST)

mkdir(TGT_ROOT_VAL)
mkdir(TGT_ROOT_TEST)

def get_class_mapping(txt_path):
    mapping = {}
    with open(txt_path, 'r') as fp:

        for i, line in enumerate(fp):
            splitted = line.split('\t')
            print(i, txt_path)
            if splitted[1] in mapping:
                mapping[splitted[1]].append(splitted[0])
            else:
                mapping[splitted[1]] = [splitted[0]]
    return mapping

def copy_by_class_mapping(mapping: dict, tgt_path, src_path):
    c = 0
    for cls, datapoints in mapping.items():
        if not exists(join(tgt_path, cls)):
            mkdir(join(tgt_path, cls))
        for dp in datapoints:
            c += 1
            if c%100 == 0:
                print(c)

            print(join(src_path, dp), join(tgt_path, dp))
            copyfile(join(src_path, dp), join(tgt_path, cls, dp))

val_mapping = get_class_mapping(join(dirname(TGT_ROOT_VAL), "val_annotations.txt"))
copy_by_class_mapping(val_mapping, TGT_ROOT_VAL, SRC_ROOT_VAL)

#tst_mapping = get_class_mapping(join(dirname(TGT_ROOT_TEST), "test_annotations.txt"))
#copy_by_class_mapping(val_mapping, TGT_ROOT_TEST, SRC_ROOT_TEST)