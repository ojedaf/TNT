# code is based on https://github.com/katerakelly/pytorch-maml
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader,Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler
import cv2
#import videotransforms
from torchvideotransforms import video_transforms, volume_transforms
from scipy.ndimage.interpolation import rotate
import json

def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    
class Rotate(object):
    def __init__(self, angle):
        self.angle = angle
    def __call__(self, x, mode="reflect"):
        x = rotate(x, self.angle, reshape=False)
        #x = x.rotate(self.angle)
        return x

def omniglot_character_folders():
    data_folder = '../datas/omniglot_resized/'

    character_folders = [os.path.join(data_folder, family, character) \
                for family in os.listdir(data_folder) \
                if os.path.isdir(os.path.join(data_folder, family)) \
                for character in os.listdir(os.path.join(data_folder, family))]
    random.seed(1)
    random.shuffle(character_folders)

    num_train = 1200
    metatrain_character_folders = character_folders[:num_train]
    metaval_character_folders = character_folders[num_train:]

    return metatrain_character_folders,metaval_character_folders

class KineticsTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, main_folder, num_classes, train_num,test_num, path_frames, num_snippets):
        
        self.path_frames = path_frames
        self.num_snippets = num_snippets
        self.main_folder = main_folder
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        list_videos_per_class = self.get_data(self.main_folder, num_classes)
        train_labels, train_roots, valid_labels, valid_roots = self.get_labels_roots(list_videos_per_class,train_num,test_num)
        
        self.roots = {'train': train_roots, 'test': valid_roots}
        self.labels = {'train': train_labels, 'test': valid_labels}
        
#     def get_data(self, path, num_classes):
#         list_classes = os.listdir(path)
#         list_videos_per_class = {}
#         for i, c in enumerate(list_classes):
#             path_c = os.path.join(path, c)
#             list_videos = os.listdir(path_c)
#             if len(list_videos)>0:
#                 list_videos_per_class[c] = []
#                 for v in list_videos:
#                     video_path = os.path.join(path_c, v)
#                     list_videos_per_class[c].append(video_path)
                    
#         new_list_classes = list(list_videos_per_class.keys())
#         new_list_classes = random.sample(new_list_classes,num_classes)
#         dict_keys = {c:list_videos_per_class[c] for c in new_list_classes}
            
#         return dict_keys

    def validate_videos_class(self, videos_per_class):
        new_vid_per_class = {}
        for i, (c, videos_c) in enumerate(videos_per_class.items()):
            list_vid = []
            for vid_name in videos_c:
                path_video = os.path.join(self.path_frames, vid_name)
                num_frames = len(os.listdir(path_video))
                if num_frames >= self.num_snippets: 
                    list_vid.append(vid_name)
            if len(list_vid) >= self.train_num + self.test_num:
                new_vid_per_class[c] = list_vid
        return new_vid_per_class
    
    def get_data(self, path, num_classes):
        with open(path, 'r') as f:
            videos_per_class = json.load(f)
        
        videos_per_class = self.validate_videos_class(videos_per_class)
        
        list_classes = list(videos_per_class.keys())
        print('Num training classes: ', len(list_classes))
        new_list_classes = random.sample(list_classes,num_classes)
        dict_keys = {c:videos_per_class[c] for c in new_list_classes}
            
        return dict_keys
    
    def get_labels_roots(self, list_videos_per_class, train_num,test_num):
        train_labels, valid_labels = [], []
        train_videos, valid_videos = [], []
        for i, (c, videos_c) in enumerate(list_videos_per_class.items()):
            videos_ran = random.sample(videos_c, len(videos_c))
            random.shuffle(videos_ran)
            train_videos.extend(videos_ran[:train_num])
            valid_videos.extend(videos_ran[train_num:train_num+test_num])
            
            train_label_c = i*np.ones(train_num)
            train_labels.extend(train_label_c)
            
            valid_label_c = i*np.ones(test_num)
            valid_labels.extend(valid_label_c)
            
        return train_labels, train_videos, valid_labels, valid_videos

class FewShotVideoDataset(Dataset):

    def __init__(self, path_frames, task, num_snippets = 20, split='train', transform=None, rotate=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.video_root = self.task.roots[split]
        self.labels = self.task.labels[split]       
        self.num_snippets = num_snippets 
        self.rotate = rotate
        self.path_frames = path_frames
        

    def __getitem__(self, idx):
        
        video_path = os.path.join(self.path_frames, self.video_root[idx])
        frames = os.listdir(video_path)
        lon_snippets = int(len(frames)/self.num_snippets)
        #values = np.linspace(0, len(frames) - 1, num=self.num_snippets)
        i = 0
        while(i<self.num_snippets):
            value = random.randint(lon_snippets*i , lon_snippets*(i+1) - 1)
        #for i,value in enumerate(values):
            frame = frames[int(value)]
            frame_path = os.path.join(video_path, frame)
            image = cv2.imread(frame_path)[:, :, [2, 1, 0]]
            w,h,c = image.shape
            if w < 226 or h < 226:
                d = 226.-min(w,h)
                sc = 1+d/min(w,h)
                image = cv2.resize(image,dsize=(0,0),fx=sc,fy=sc)
            image = (image/255.)*2 - 1
            if i == 0:
                video_frames = np.zeros((self.num_snippets,) + image.shape, np.float32)
            if self.rotate is not None:
                print('Rotate video')
                image = self.rotate(image)
            video_frames[i, :, :, :] = image
            i += 1
            
        #image = image.resize((28,28), resample=Image.LANCZOS) # per Chelsea's implementation
        if self.transform is not None:
            video_frames = self.transform(video_frames)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        #video_frames = torch.from_numpy(video_frames.transpose([3,0,1,2]))
        return video_frames, torch.from_numpy(np.array([label]))

    def __len__(self):
        return len(self.video_root)



class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L')
        image = image.resize((28,28), resample=Image.LANCZOS) # per Chelsea's implementation
        #image = np.array(image, dtype=np.float32)
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=1, split='train',shuffle=True,rotation=0):
    # NOTE: batch size here is # instances PER CLASS
    #normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    #normalize = transforms.Normalize(mean=[0.92206,], std=[0.08426,]) 
    # transforms_video_train = transforms.Compose([
    #             videotransforms.RandomCrop(100)])


    video_transform_list_train = [video_transforms.Resize((256, 256)),
			video_transforms.RandomCrop((224, 224)),
            # video_transforms.Resize((100, 100)),
            video_transforms.RandomHorizontalFlip(),
			volume_transforms.ClipToTensor()]

    video_transform_list_test = [video_transforms.Resize((256, 256)),
			video_transforms.CenterCrop((224, 224)),
            # video_transforms.Resize((100, 100)),
			volume_transforms.ClipToTensor()]

    if split == 'train':
        transforms_train = video_transforms.Compose(video_transform_list_train)
        dataset = FewShotVideoDataset(task.path_frames,task,num_snippets = task.num_snippets, split=split,transform = transforms_train)
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        transforms_test = video_transforms.Compose(video_transform_list_test)
        dataset = FewShotVideoDataset(task.path_frames,task,num_snippets = task.num_snippets, split=split,transform = transforms_test)
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler, pin_memory=True)

    return loader
