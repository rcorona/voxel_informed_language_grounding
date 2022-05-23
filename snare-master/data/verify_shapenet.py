import os
import json
import pdb

# Get all objects that exist in SNARE. 
def get_snare_objs(split_sets=False): 
    
    snare_path = '/home/rcorona/obj_part_lang/snare-master/amt/folds_adversarial'
    train = json.load(open(os.path.join(snare_path, 'train.json')))
    val = json.load(open(os.path.join(snare_path, 'val.json')))
    test = json.load(open(os.path.join(snare_path, 'test.json')))

    train_objs = set()
    val_objs = set()
    test_objs = set()

    # Comb through snare files to collect unique set of ShapeNet objects. 
    snare_objs = set()

    for obj_set, split in [(train_objs, train), (val_objs, val), (test_objs, test)]:
        for datapoint in split: 
            for obj in datapoint['objects']:
                obj_set.add(obj)

    all_objs = train_objs | val_objs | test_objs

    if not split_sets: 
        return all_objs
    else:
        return (train_objs, val_objs, test_objs)

if __name__ == '__main__':
    voxel_paths = dict()
    voxel_parent_dir = '/home/rcorona/obj_part_lang/snare-master/data/models-binvox-solid'

    # Get all objects for which we have gt voxel annotations. 
    for obj_file in os.listdir(voxel_parent_dir): 

        obj_path = os.path.join(voxel_parent_dir, obj_file)

        if os.path.isfile(obj_path): 
            obj = obj_file.split('.')[0]
            voxel_paths[obj] = obj_path

    snare_objs = get_snare_objs()
