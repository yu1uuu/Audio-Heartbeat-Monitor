import os
import shutil
import random

def split_datasets(src_path, dest_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    '''split dataset into train, val, test sets with 0.7 / 0.15 / 0.15 ratio'''

    # make new directory for each dataset if does not exist
    for dataset in ['Train', 'Val', 'Test']:
        for class_folder in ['Normal', 'Irregular']:
            os.makedirs(os.path.join(dest_path, dataset, class_folder), exist_ok=True)

    # get folder of each heartbeat class
    for class_folder in ['Normal', 'Irregular']:
        class_path = os.path.join(src_path, class_folder)
        files = os.listdir(class_path)
        random.shuffle(files) # shuffle files for a random split

        total_files = len(files)
        train_end = int(total_files * train_ratio) # end of train set
        val_end = train_end + int(total_files * val_ratio) # end of val set

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # split the dataset into the Train, Val, and Test folders
        for file in train_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(dest_path, 'Train', class_folder))
        for file in val_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(dest_path, 'Val', class_folder))
        for file in test_files:
            shutil.copy(os.path.join(class_path, file), os.path.join(dest_path, 'Test', class_folder))

src_directory = '' # source directory path here
dest_directory = '' # destination directory path here
split_datasets(src_directory, dest_directory)
