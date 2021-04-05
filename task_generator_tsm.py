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
import json, pickle
from numpy.random import randint
from multiprocessing.dummy import Pool

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

def validate_videos_class(videos_per_class, path_frames, num_snippets, train_num, test_num, type_text, type_dataset):
        new_vid_per_class = {}
        num_vid_total = 0
        for i, (c, videos_c) in enumerate(videos_per_class.items()):
            list_vid = []
            for vid_obj in videos_c:
                if type(vid_obj) == dict:
                    vid_name = vid_obj['id']
                else:
                    vid_name = vid_obj
                path_video = os.path.join(path_frames, vid_name)
                if type_dataset == 'epicKitchens':
                    frames = vid_obj['frames']
                else:
                    frames = os.listdir(path_video)
                    frames = [f for f in frames if f.endswith(('.jpg', '.png'))]
                num_frames = len(frames)
                if num_frames >= num_snippets: 
                    if type_text == 'label':
                        text = vid_obj['label']
                    else:
                        text = c
                    text = text.lower().replace('(','').replace(')','').replace('[','').replace(']','').replace('kercief','kerchief')
                    if type_dataset == 'epicKitchens':
                        list_vid.append({'video': vid_name, 'label': text, 'start_frame': vid_obj['start_frame'], 'stop_frame': vid_obj['stop_frame'], 'frames': vid_obj['frames']})
                    else:
                        list_vid.append({'video': vid_name, 'label': text})
            if len(list_vid) >= train_num + test_num:
                new_vid_per_class[c] = list_vid
                num_vid_total += len(list_vid)
        return new_vid_per_class, num_vid_total

def get_data_classes(path, path_frames, num_snippets, train_num, test_num, type_text, type_dataset):
    try:
        with open(path, 'r') as f:
            videos_per_class = json.load(f)
    except:
        with open(path, 'rb') as f:
            videos_per_class = pickle.load(f)
    videos_per_class, num_vid_total = validate_videos_class(videos_per_class, path_frames, num_snippets, train_num, test_num, type_text, type_dataset)
    return videos_per_class, num_vid_total


class S2Sv2Task(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, videos_per_class, num_classes, train_num,test_num, path_frames, num_snippets, num_episodes):
        
        self.path_frames = path_frames
        self.num_snippets = num_snippets
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        train_labels_by_episode = []
        train_roots_by_episode = []
        valid_labels_by_episode = []
        valid_roots_by_episode = []
        for i in range(num_episodes):
            list_videos_per_class = self.get_data(videos_per_class, num_classes)
            train_labels, train_roots, valid_labels, valid_roots = self.get_labels_roots(list_videos_per_class,train_num,test_num)
            train_labels_by_episode.append(train_labels)
            train_roots_by_episode.append(train_roots)
            valid_labels_by_episode.append(valid_labels)
            valid_roots_by_episode.append(valid_roots)

        self.num_vid = len(train_labels_by_episode[0]) + len(valid_labels_by_episode[0])
        
        self.roots = {'train': train_roots_by_episode, 'test': valid_roots_by_episode}
        self.labels = {'train': train_labels_by_episode, 'test': valid_labels_by_episode}

    # def validate_videos_class(self, videos_per_class):
    #     new_vid_per_class = {}
    #     num_vid_total = 0
    #     for i, (c, videos_c) in enumerate(videos_per_class.items()):
    #         list_vid = []
    #         for vid_obj in videos_c:
    #             vid_name = vid_obj['id']
    #             path_video = os.path.join(self.path_frames, vid_name)
    #             num_frames = len(os.listdir(path_video))
    #             if num_frames >= self.num_snippets: 
    #                 list_vid.append(vid_name)
    #         if len(list_vid) >= self.train_num + self.test_num:
    #             new_vid_per_class[c] = list_vid
    #             num_vid_total += len(list_vid)
    #     return new_vid_per_class, num_vid_total
    
    # def get_data(self, path, num_classes):
    #     with open(path, 'r') as f:
    #         videos_per_class = json.load(f)
        
    #     videos_per_class, num_vid_total = self.validate_videos_class(videos_per_class)
        
    #     list_classes = list(videos_per_class.keys())
    #     # print('Num training classes: ', len(list_classes))
    #     new_list_classes = random.sample(list_classes,num_classes)
    #     # print("new_list_classes: ",new_list_classes)
    #     dict_keys = {c:videos_per_class[c] for c in new_list_classes}
            
    #     return dict_keys, num_vid_total

    def get_data(self, videos_per_class, num_classes):
        list_classes = list(videos_per_class.keys())
        # print('Num training classes: ', len(list_classes))
        new_list_classes = random.sample(list_classes,num_classes)
        # print("new_list_classes: ",new_list_classes)
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

class KineticsTask(object):
    # This class is for task generation for both meta training and meta testing.
    # For meta training, we use all 20 samples without valid set (empty here).
    # For meta testing, we use 1 or 5 shot samples for training, while using the same number of samples for validation.
    # If set num_samples = 20 and chracter_folders = metatrain_character_folders, we generate tasks for meta training
    # If set num_samples = 1 or 5 and chracter_folders = metatest_chracter_folders, we generate tasks for meta testing
    def __init__(self, videos_per_class, num_classes, train_num,test_num, path_frames, num_snippets, num_episodes):
        
        self.path_frames = path_frames
        self.num_snippets = num_snippets
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = test_num
        train_labels_by_episode = []
        train_roots_by_episode = []
        valid_labels_by_episode = []
        valid_roots_by_episode = []
        for i in range(num_episodes):
            list_videos_per_class = self.get_data(videos_per_class, num_classes)
            train_labels, train_roots, valid_labels, valid_roots = self.get_labels_roots(list_videos_per_class,train_num,test_num)
            train_labels_by_episode.append(train_labels)
            train_roots_by_episode.append(train_roots)
            valid_labels_by_episode.append(valid_labels)
            valid_roots_by_episode.append(valid_roots)
        
        self.num_vid = len(train_labels_by_episode[0]) + len(valid_labels_by_episode[0])

        self.roots = {'train': train_roots_by_episode, 'test': valid_roots_by_episode}
        self.labels = {'train': train_labels_by_episode, 'test': valid_labels_by_episode}
        
    def get_data(self, videos_per_class, num_classes):
        list_classes = list(videos_per_class.keys())
        # print('Num training classes: ', len(list_classes))
        new_list_classes = random.sample(list_classes,num_classes)
        # print("new_list_classes: ",new_list_classes)
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

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def frames(self):
        return self._data['frames']

    @property
    def num_frames(self):
        return len(self._data['frames'])

    @property
    def path_video(self):
        return self._data['path_video']

    @property
    def label(self):
        return int(self._data['label'])

class FewShotVideoDatasetV2(Dataset):

    def __init__(self, root_path, task, type_dataset, split='train',
                 num_segments=3, new_length=1, modality='RGB',
                 transform=None, random_shift=True, test_mode=False,
                 remove_missing=False, dense_sample=False, twice_sample=False, workers = None, shuffle=True):

        self.root_path = root_path
        self.type_dataset = type_dataset
        self.task = task
        self.split = split
        self.video_root = self.task.roots[split]
        self.labels = self.task.labels[split]   
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self.pool = workers         
        self.shuffle = shuffle
        
        # self._parse_list()

    # def _parse_list(self):
    #     self.video_list = list()
    #     for i, path_vid in enumerate(self.video_root):
    #         video_path = os.path.join(self.root_path, path_vid)
    #         frames = os.listdir(video_path)
    #         label = self.labels[i]
    #         self.video_list.append(VideoRecord({'frames': frames,'label': label, 'path_video': video_path}))
    #     print('video number:%d' % (len(self.video_list)))

    def _load_image(self, frame_path, path_video):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            try:
                return [Image.open(os.path.join(path_video, frame_path)).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(path_video, frame_path))
                return [Image.open(os.path.join(path_video, frame_path)).convert('RGB')]
        elif self.modality == 'Flow':
            if self.image_tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(path_video, frame_path)).convert(
                    'L')
                y_img = Image.open(os.path.join(path_video, frame_path)).convert(
                    'L')
            elif self.image_tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(path_video, '{:06d}'.format(int(frame_path)))).convert('L')
                y_img = Image.open(os.path.join(path_video, '{:06d}'.format(int(frame_path)))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(self.root_path, frame_path)).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(self.root_path, frame_path))
                    flow = Image.open(os.path.join(self.root_path, frame_path)).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:  # normal sample
            average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record.num_frames > self.num_segments:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        else:
            if record.num_frames > self.num_segments + self.new_length - 1:
                tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return offsets + 1

    def _get_test_indices(self, record):
        if self.dense_sample:
            sample_pos = max(1, 1 + record.num_frames - 64)
            t_stride = 64 // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=10, dtype=int)
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record.num_frames for idx in range(self.num_segments)]
            return np.array(offsets) + 1
        elif self.twice_sample:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return offsets + 1
        else:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return offsets + 1

    def get_elem(self, index):
        vid_obj = self.episode_path_vid[index]
        path_vid = vid_obj['video']
        label_text = vid_obj['label']
        label = self.episode_label[index]

        video_path = os.path.join(self.root_path, path_vid)

        if self.type_dataset == 'epicKitchens':
            frames = vid_obj['frames']
            frames_num = [i.split('.')[0].split('_')[1] for i in frames]
        else:
            frames = os.listdir(video_path)
            frames = [f for f in frames if f.endswith(('.jpg', '.png'))]
            frames_num = [i.split('.')[0] for i in frames]
        idxs = sorted(range(len(frames_num)), key=lambda k: float(frames_num[k]))
        frames = [frames[i] for i in idxs]

        record = VideoRecord({'frames': frames,'label': label, 'path_video': video_path})

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        
        img, label = self.get(record, segment_indices)
        return img, label, label_text

    def __getitem__(self, index):
        self.episode_path_vid = self.video_root[index]
        self.episode_label = self.labels[index]

        list_img_episode = self.pool.map(self.get_elem, list(range(len(self.episode_label))))
        if self.shuffle:
            random.shuffle(list_img_episode)

        imgs_episode, labels_episode, labels_text_episode = zip(*list_img_episode)
        imgs_episode = torch.stack(imgs_episode, dim = 0)
        labels_episode = torch.tensor(labels_episode)
        return imgs_episode, labels_episode, list(labels_text_episode)
        

    def get(self, record, indices):

        images = list()
        frames = record.frames
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(frames[p-1],record.path_video)
                images.extend(seg_imgs)

                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_root)



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


def get_data_loader(task, type_dataset, num_per_class=1, split='train',shuffle=True,rotation=0, transforms = None, test_mode = False, workers = None):
    # NOTE: batch size here is # instances PER CLASS
    #normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
    #normalize = transforms.Normalize(mean=[0.92206,], std=[0.08426,]) 
    # transforms_video_train = transforms.Compose([
    #             videotransforms.RandomCrop(100)])

    # video_transform_list_train = [video_transforms.Resize((256, 256)),
	# 		video_transforms.RandomCrop((224, 224)),
    #         # video_transforms.Resize((100, 100)),
    #         video_transforms.RandomHorizontalFlip(),
	# 		volume_transforms.ClipToTensor()]

    # video_transform_list_test = [video_transforms.Resize((256, 256)),
	# 		video_transforms.CenterCrop((224, 224)),
    #         # video_transforms.Resize((100, 100)),
	# 		volume_transforms.ClipToTensor()]

    if split == 'train':

        #transforms_train = video_transforms.Compose(video_transform_list_train)
        transforms_train = torchvision.transforms.Compose(transforms)

        dataset = FewShotVideoDatasetV2(task.path_frames, task, type_dataset = type_dataset, split=split, num_segments=task.num_snippets,
                   new_length=1,
                   modality='RGB',
                   test_mode=test_mode,
                   transform=transforms_train, dense_sample=False, workers = workers, shuffle=shuffle)

        # dataset = FewShotVideoDataset(task.path_frames,task,num_snippets = task.num_snippets, split=split,transform = transforms_train)
        # sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num,shuffle=shuffle)
    else:
        # transforms_test = video_transforms.Compose(video_transform_list_test)
        transforms_test = torchvision.transforms.Compose(transforms)
        # dataset = FewShotVideoDataset(task.path_frames,task,num_snippets = task.num_snippets, split=split,transform = transforms_test)

        dataset = FewShotVideoDatasetV2(task.path_frames, task, type_dataset = type_dataset, split=split, num_segments=task.num_snippets, 
                    new_length=1,
                    modality = "RGB",
                    test_mode=test_mode,
                    transform=transforms_test, dense_sample=False, workers = workers, shuffle=shuffle)
                    
        # sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num,shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=1, pin_memory=True)

    return loader
