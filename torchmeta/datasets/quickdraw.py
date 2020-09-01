import os
import pickle
from PIL import Image
import h5py
import json
import numpy as np

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import download_file_from_google_drive


class QuickDraw(CombinationMetaDataset):
    """
    The QuickDraw dataset, contains images of 345 different classes. 
    The meta train/validation/test splits are 60:20:20.

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `quickdraw-dataset` exists.

    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way" 
        classification.

    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the 
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the 
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`, 
        `meta_val` and `meta_test` if all three are set to `False`.

    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed 
        version. See also `torchvision.transforms`.

    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed 
        version. See also `torchvision.transforms`.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a 
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes 
        are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.

    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root 
        directory (under the `quickdraw-dataset` folder). If the dataset is already 
        available, this does not download/process the dataset again.
        
    random_seed : int (default: 123)
        Sets random seed to select train, val and test classes
        
    num_training_samples : int (default: 100)
        Down samples the QuickDraw dataset to include approx. user specified number of 
        samples for each class

    Notes
    -----
    The dataset is downloaded from [this repository]
    (https://github.com/googlecreativelab/quickdraw-dataset). The meta train/
    validation/test splits are over 60/20/20 classes.

    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False, random_seed=123, num_training_samples=100):
        dataset = QuickDrawClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download, random_seed=random_seed, num_training_samples=num_training_samples)
        super(QuickDraw, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class QuickDrawClassDataset(ClassDataset):
    folder = 'quickdraw-dataset'
    #folder_meta = 'quickdraw-dataset-meta' # whole
    #folder_meta = 'quickdraw-sample-meta' # sample
    folder_meta = 'quickdraw-{0}-meta' # 100 or 20 samples, with random class split
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'
    train_val_test_ratio = [60, 20, 20]

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False, random_seed=123, num_training_samples=100):
        super(QuickDrawClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
        self.random_seed = random_seed
        self.num_training_samples=num_training_samples
        self.root = os.path.join(os.path.expanduser(root)) #, self.folder
        self.transform = transform
    
        self.split_filename = os.path.join(self.root, self.folder_meta.format(self.num_training_samples), 
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root, self.folder_meta.format(self.num_training_samples), 
            self.filename_labels.format(self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('QuickDraw integrity check failed')
        self._num_classes = len(self.labels)


    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)
        return QuickDrawDataset(index, data, class_name,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        if self._check_integrity():
            return
        
        if not os.path.exists(os.path.join(self.root, self.folder)): 
            print("Downloading data in ", self.root)
            cmd = "gsutil -m cp -r gs://quickdraw_dataset/full/numpy_bitmap/*.npy {0}".format(os.path.join(self.root, self.folder))
            os.system(cmd)

        foldername = os.path.join(self.root, self.folder_meta.format(self.num_training_samples))
        if os.path.exists(foldername):
            return
        
        os.mkdir(foldername)                
        filenames = os.listdir(os.path.join(self.root, self.folder))
        classes = sorted(filenames)
        # remove "*.npy" from label names
        classes = [sub[: -4] for sub in classes]
        # shuffle classes so that the train, val and test classes are selected at random
        rs = np.random.RandomState(self.random_seed)
        rs.shuffle(classes)
        print(f'Total classes: {len(classes)}')
        print(classes)
        num_class = len(classes)

        num_train, num_val, num_test = [int(float(ratio)/np.sum(self.train_val_test_ratio)*num_class)
                                        for ratio in self.train_val_test_ratio]
        for split in ['train', 'val', 'test']:
            
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue
            
            labels_filename = os.path.join(self.root,
                                           self.folder_meta.format(self.num_training_samples), 
                                           self.filename_labels.format(split))
            labels = []
            with open(labels_filename, 'w') as f:
                if split == 'train':
                    labels = classes[:num_train]     
                elif split == 'val':
                    labels = classes[num_train:num_train+num_val]
                else:
                    labels = classes[num_train+num_val:]
                json.dump(labels, f)
            
            filename = os.path.join(self.root, 
                                    self.folder_meta.format(self.num_training_samples),
                                    self.filename.format(split))                
            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                for classname in labels:
                 
                    data = np.load(os.path.join(self.root, 
                                                self.folder, 
                                                classname+'.npy')) #+'.npy'
                    print(data.shape)
                    # This is not exact but creates approx 100 or 20 samples
                    sample = data[rs.choice(data.shape[0], self.num_training_samples, replace=False)]
                    '''
                    mask = rs.choice([False, True], len(data), 
                                     p=[1.0-(self.num_training_samples/len(data)),
                                        self.num_training_samples/len(data)])
                    data = data[mask]
                    '''
                    print('number of samples in {} after down-sampling: {}'.format(classname, sample.shape[0]))
                    sample = sample.reshape((sample.shape[0], 28, 28))
                    #print(data.shape)               
                    group.create_dataset(classname, data=sample)

class QuickDrawDataset(Dataset):
    def __init__(self, index, data, class_name,
                 transform=None, target_transform=None):
        super(QuickDrawDataset, self).__init__(index, transform=transform,
                                                  target_transform=target_transform)
        self.data = data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index], mode='L')
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
