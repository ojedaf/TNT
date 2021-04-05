
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator_tsm as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
#from model.rnact_tsm import CNNEncoder,RelationNetwork,CovpoolLayer,PowerNorm, AttModule
from model.tsm_cnaps_model import SimpleTSMCnaps
from model.utils import print_and_log, get_log_files, ValidationAccuracies, loss, aggregate_accuracy
import statistics 
import yaml
import time
from model.temporalShiftModule.ops.transforms import *
from multiprocessing.dummy import Pool

def parse_conf(conf, new_dict = {}):
    for k, v in conf.items():
        if type(v) == dict:
            new_dict = parse_conf(v, new_dict)
        else:
            new_dict[k] = v
    return new_dict

parser = argparse.ArgumentParser(description="Few Shot Video Recognition")
parser.add_argument("-conf","--conf_path", default = './conf/conf_base_tsm_cnaps.yaml')
args = parser.parse_args()
conf_file = open(args.conf_path, 'r')
print("Conf file dir: ",conf_file)
dict_conf = yaml.load(conf_file)

# Hyper Parameters
NUM_TASK_PER_BATCH = dict_conf['num_task_per_batch']
distance = dict_conf['distance_metric']
feature_adaptation = dict_conf['feature_adaptation']
train_feature_encoder = dict_conf['train_feature_encoder']
CLASS_NUM = dict_conf['class_num']
SAMPLE_NUM_PER_CLASS = dict_conf['sample_num_per_class']
BATCH_NUM_PER_CLASS = dict_conf['batch_num_per_class']
EPISODE = dict_conf['train_episode']
num_episode_decay = dict_conf['num_episode_decay']
VAL_EPISODE = dict_conf['val_episode']
num_episode_to_val = dict_conf['num_episode_to_val']
TEST_EPISODE = dict_conf['test_episode']
lr_cnaps = dict_conf['learning_rate_cnaps']
api_key = dict_conf['comet']['api_key']
workspace = dict_conf['comet']['workspace']
project_name = dict_conf['comet']['project_name']
conf_tsm = dict_conf['TSM']
experiment = Experiment(api_key=api_key,
                        project_name=project_name, workspace=workspace)

experiment.log_parameters(parse_conf(dict_conf))

dataset_name = dict_conf['dataset']['name']
path_metatrain = dict_conf['dataset']['path_metatrain']
path_metaval = dict_conf['dataset']['path_metaval']
path_metatest = dict_conf['dataset']['path_metatest']
path_frames = dict_conf['dataset']['path_frames']
path_checkpoint_feature_encoder = dict_conf['checkpoints']['path_feature_encoder']
path_checkpoint_model = dict_conf['checkpoints']['path_model']
pooling = conf_tsm['pooling']
text_type = dict_conf['text']['type']
test_1shot_from_5shot = dict_conf['test_1shot_from_5shot'] if 'test_1shot_from_5shot' in dict_conf else False

pool = Pool(processes=10) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_accuracy(logits, label):
    pred = torch.argmax(logits.cpu(), dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean()
    return accuracy.item()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m,h

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

def adjust_learning_rate(optimizer, epoch, lr_type, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if lr_type == 'step':
        decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
        lr = conf_tsm['lr'] * decay
        decay = conf_tsm['weight_decay']
    elif lr_type == 'cos':
        import math
        lr = 0.5 * conf_tsm['lr'] * (1 + math.cos(math.pi * epoch / conf_tsm['epochs']))
        decay = conf_tsm['weight_decay']
    else:
        raise NotImplementedError
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def test(model, transform_test, meta_data, episodes, taskGen, print_enable = False):
    print("Testing... N_episode: {}".format(episodes), flush=True)
    accuracies = []

    with torch.no_grad():

        task_val = taskGen(meta_data, CLASS_NUM, SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS, path_frames, num_snippets = conf_tsm['num_segments'], num_episodes = episodes)
        sample_dataloader_val = tg.get_data_loader(task_val, type_dataset = dataset_name, num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False, transforms = transform_test,  test_mode = True, workers = pool)
        batch_dataloader_val = tg.get_data_loader(task_val, type_dataset = dataset_name, num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True, transforms = transform_test,  test_mode = True, workers = pool)
        it_sampleloader_val = iter(sample_dataloader_val)
        it_batchloader_val = iter(batch_dataloader_val)
        
        for i in range(episodes):
            
            sample_images,sample_labels,sample_labels_text = next(it_sampleloader_val)
            sample_images,sample_labels,sample_labels_text = torch.squeeze(sample_images, dim =0),torch.squeeze(sample_labels, dim = 0), list(np.array(sample_labels_text)[:,0])
            batches,batch_labels,_ = next(it_batchloader_val)
            batches,batch_labels = torch.squeeze(batches, dim =0),torch.squeeze(batch_labels, dim = 0)

            batch_labels = batch_labels.type(torch.LongTensor)

            target_logits = model(support_videos = sample_images, support_labels = sample_labels, support_label_text = sample_labels_text, query_videos = batches)

            if distance == 'mahalanobis':
                task_accuracy = aggregate_accuracy(target_logits.cpu(), batch_labels) * 100.0
            else:
                task_accuracy = count_accuracy(target_logits, batch_labels)
            accuracies.append(task_accuracy)

        mean_accuracy, h = mean_confidence_interval(accuracies)

        return mean_accuracy, h

def train(type_loss, model, optimizer, feature_enco_optim, transform_train, transform_test, taskGen, metatrain_data, metaval_data, last_saved_accuracy = 0):

    print("Training...", flush=True)

    last_accuracy = last_saved_accuracy
    print("Last accuracy: ", last_accuracy)


    print('Distance: ', distance)
    if distance != 'mahalanobis':
        if type_loss == 'CrossEntropyLoss':
            loss_fn = nn.CrossEntropyLoss().to(device)
        else:
            loss_fn = nn.MSELoss().to(device)

    with experiment.train():

        task = taskGen(metatrain_data, CLASS_NUM, SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS, path_frames, num_snippets = conf_tsm['num_segments'], num_episodes = EPISODE)
    
        sample_dataloader = tg.get_data_loader(task, type_dataset = dataset_name, num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False, transforms = transform_train,  test_mode = False, workers = pool)
        batch_dataloader = tg.get_data_loader(task, type_dataset = dataset_name, num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True, transforms = transform_train,  test_mode = False, workers = pool)
        it_sampleloader = iter(sample_dataloader)
        it_batchloader = iter(batch_dataloader)
        
        for episode in range(EPISODE):
            torch.set_grad_enabled(True)
            if train_feature_encoder:
                model.feature_extractor.train()
                if conf_tsm['no_partialbn']:
                    if torch.cuda.device_count() > 1:
                        model.feature_extractor.module.partialBN(False)
                    else:
                        model.feature_extractor.partialBN(False)
                else:
                    if torch.cuda.device_count() > 1:
                        model.feature_extractor.module.partialBN(True)
                    else:
                        model.feature_extractor.partialBN(True)
            
            # sample datas
            samples,sample_labels,sample_labels_text = next(it_sampleloader) #25*3*84*84
            samples,sample_labels,sample_labels_text = torch.squeeze(samples, dim =0),torch.squeeze(sample_labels, dim = 0), list(np.array(sample_labels_text)[:,0])
            batches,batch_labels,_ = next(it_batchloader)
            batches,batch_labels = torch.squeeze(batches, dim =0),torch.squeeze(batch_labels, dim = 0)
        
            batch_labels = batch_labels.type(torch.LongTensor)

            target_logits = model(support_videos = samples, support_labels = sample_labels, support_label_text = sample_labels_text, query_videos = batches)
            
            if distance == 'mahalanobis':
                task_loss = loss(target_logits, batch_labels, device)
                if feature_adaptation == 'film' or feature_adaptation == 'film+ar':
                    if torch.cuda.device_count() > 1:
                        regularization_term = (model.feature_adaptation_network.module.regularization_term())
                    else:
                        regularization_term = (model.feature_adaptation_network.regularization_term())
                    regularizer_scaling = 0.001
                    task_loss += regularizer_scaling * regularization_term
                task_accuracy = aggregate_accuracy(target_logits.cpu(), batch_labels) * 100.0           
            else: 
                if type_loss =='MSELoss': 
                    one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM, dtype=torch.long).scatter_(1, batch_labels.view(-1,1), 1).to(device))
                else:
                    one_hot_labels = batch_labels.to(device)

                task_loss = loss_fn(target_logits,one_hot_labels)
                task_accuracy = count_accuracy(target_logits, batch_labels)
            
            task_loss.backward(retain_graph=False)
            # training
            if ((episode + 1) % NUM_TASK_PER_BATCH == 0) or (episode == (EPISODE - 1)):
                optimizer.step()
                optimizer.zero_grad()

                if train_feature_encoder == True:
                    feature_enco_optim.step()
                    feature_enco_optim.zero_grad()


            experiment.log_metric("accuracy", task_accuracy, step=episode)
            experiment.log_metric("loss", task_loss.item(), step=episode)

            if (episode + 1)%num_episode_to_val == 0 or episode == 0:
                # test
                with experiment.validate():

                    if train_feature_encoder:
                        model.feature_extractor.eval()
                        
                    test_accuracy, h = test(model, transform_test, metaval_data, VAL_EPISODE, taskGen, False)
                    print("test Accuracy episode {}: {:.2f} ± {:.2f} %".format(episode, test_accuracy,h), flush=True)
                    experiment.log_metric("accuracy", test_accuracy, step = episode)
                    experiment.log_metric("confidence interval", h, step = episode)

                    if test_accuracy > last_accuracy:
                        # save networks
                        print('Saving ... ')
                        torch.save(model.state_dict(),str(path_checkpoint_model + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                        print("save networks for episode:",episode, flush=True)
                        last_accuracy = test_accuracy



def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction

    # Step 2: init neural networks
    print("init neural networks", flush=True)

    model = SimpleTSMCnaps(device, conf = dict_conf)

    # input_size = feature_encoder.TSM.input_size
    crop_size = model.feature_extractor.crop_size
    scale_size = model.feature_extractor.scale_size
    input_mean = model.feature_extractor.input_mean
    input_std = model.feature_extractor.input_std
    policies = model.feature_extractor.get_optim_policies()

    if conf_tsm['test_crops'] == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(crop_size),
        ])
    elif conf_tsm['test_crops'] == 3:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(crop_size, scale_size, flip=False)
        ])
    elif conf_tsm['test_crops'] == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(crop_size, scale_size, flip=False)
        ])
    elif conf_tsm['test_crops'] == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(crop_size, scale_size)
        ])
    else:
        raise ValueError("Only 1, 5, 10 crops are supported while we got {}".format(args.test_crops))

    train_augmentation = model.feature_extractor.get_augmentation(flip=False if 'something' in dataset_name or 'jester' in dataset_name or 'epicKitchens' == dataset_name else True)
    taskGen = tg.S2Sv2Task if 'something' in dataset_name or 'epicKitchens' == dataset_name else tg.KineticsTask

    normalize = GroupNormalize(input_mean, input_std)
    transform_train = [
            train_augmentation,
            Stack(roll=(conf_tsm['arch'] in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(conf_tsm['arch'] not in ['BNInception', 'InceptionV3'])),
            normalize
            ]

    transform_test = [
            cropping,
            Stack(roll=(conf_tsm['arch'] in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(conf_tsm['arch'] not in ['BNInception', 'InceptionV3'])),
            normalize,
            ]
    
    print('Train feature encoder: ',train_feature_encoder)
    if train_feature_encoder == True: 
        model.train()  # set encoder is always in train mode to process context data
        model.distribute_model()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_cnaps)
        optimizer.zero_grad()

        for param in model.feature_extractor.parameters():
            param.requires_grad = True
        feature_enco_optim = torch.optim.SGD(policies, conf_tsm['lr'], momentum=conf_tsm['momentum'], weight_decay=conf_tsm['weight_decay'])
        feature_enco_optim.zero_grad()
        # feature_enco_scheduler = StepLR(feature_enco_optim,step_size=num_episode_decay,gamma=0.1)
        # Implementar optimizador del modelo y feature enconder teniendo en cuenta la politicas de este
    else:
        model.train()  # set encoder is always in train mode to process context data
        model.feature_extractor.eval()  # feature extractor is always in eval mode
        model.distribute_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_cnaps)
        optimizer.zero_grad()
        
        feature_enco_optim = None

    type_loss = 'CrossEntropyLoss'

    # if conf_tsm['pooling'] == 'attention':
    #     t_dim = conf_tsm['num_segments']
    #     radio = conf_tsm['ratio']
    #     chann_dim = conf_tsm['chann_dim']
    #     att_module = AttModule(radio, t_dim, chann_dim)
    #     att_module.to(device)

    #     att_module_optim = torch.optim.SGD(att_module.parameters(), conf_tsm['lr'], momentum=conf_tsm['momentum'], weight_decay=conf_tsm['weight_decay'])
    #     att_module_scheduler = StepLR(att_module_optim,step_size=num_episode_decay,gamma=0.1)
    # else:
    #     att_module = None
    #     att_module_optim = None
    #     att_module_scheduler = None   


    if dict_conf['train_mode'] and test_1shot_from_5shot != True:
        
        last_accuracy = 0
        if os.path.exists(str(path_checkpoint_model + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
            model.load_state_dict(torch.load(str(path_checkpoint_model + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
            print("load parameters model - to train")
            model.train()  # set encoder is always in train mode to process context data
            model.feature_extractor.eval()  # feature extractor is always in eval mode

            last_accuracy = dict_conf['last_accuracy']

        metatrain_data, num_vid_train = tg.get_data_classes(path_metatrain, path_frames, conf_tsm['num_segments'], SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS, text_type, dataset_name)
        metaval_data, num_vid_val = tg.get_data_classes(path_metaval, path_frames, conf_tsm['num_segments'], SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS, text_type, dataset_name)
        print("num_vid_train: {}, num_vid_val: {}".format(num_vid_train, num_vid_val))
        train(type_loss, model, optimizer, feature_enco_optim, transform_train, transform_test, taskGen, metatrain_data, metaval_data, last_accuracy)
    
    with experiment.test():
        print("Init test")
        if test_1shot_from_5shot:
            print("test 1-shot from 5-shot model")
            checkpoint_path = str(path_checkpoint_model + str(CLASS_NUM) +"way_" + str(5) +"shot.pkl")
        else:
            checkpoint_path = str(path_checkpoint_model + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")
        if os.path.exists(checkpoint_path):
            # dict_param = torch.load(checkpoint_path)
            # model.load_state_dict(dict_param)
            # print("load parameters model - to test: ", checkpoint_path)
            if dict_conf['train_mode'] and test_1shot_from_5shot != True:
                dict_param = torch.load(checkpoint_path)
                model.load_state_dict(dict_param)
            else:
                dict_param = torch.load(checkpoint_path, map_location=lambda device, loc: device)
                if 'module' in list(dict_param.keys())[0]:
                    new_dict_param = {}
                    for key, val in dict_param.items():
                        new_dict_param[key.replace('module.', '')] = val
                    dict_param = new_dict_param
                model.load_state_dict(dict_param)
            print("load parameters model - to test: ", checkpoint_path)

        model.train()  # set encoder is always in train mode to process context data
        model.feature_extractor.eval()  # feature extractor is always in eval mode

        metatest_data, num_vid_test = tg.get_data_classes(path_metatest, path_frames, conf_tsm['num_segments'], SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS, text_type, dataset_name)
        mean_accuracy, h = test(model, transform_test, metatest_data, TEST_EPISODE, taskGen, True)
        print("test Accuracy: {:.2f} ± {:.2f} %".format(mean_accuracy,h), flush=True)
        experiment.log_metric("accuracy", mean_accuracy)
        experiment.log_metric("confidence interval", h)
                




if __name__ == '__main__':
    main()
