
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
from model.rnact_tsm import CNNEncoder,RelationNetwork,CovpoolLayer,PowerNorm, AttModule
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
parser.add_argument("-conf","--conf_path", default = './conf/conf_base_tsm.yaml')
args = parser.parse_args()
conf_file = open(args.conf_path, 'r')
print("Conf file dir: ",conf_file)
dict_conf = yaml.load(conf_file)

# Hyper Parameters
FEATURE_DIM = dict_conf['feature_dim']
RELATION_DIM = dict_conf['relation_dim']
CLASS_NUM = dict_conf['class_num']
SAMPLE_NUM_PER_CLASS = dict_conf['sample_num_per_class']
BATCH_NUM_PER_CLASS = dict_conf['batch_num_per_class']
EPISODE = dict_conf['train_episode']
num_episode_decay = dict_conf['num_episode_decay']
VAL_EPISODE = dict_conf['val_episode']
num_episode_to_val = dict_conf['num_episode_to_val']
TEST_EPISODE = dict_conf['test_episode']
LEARNING_RATE = dict_conf['learning_rate']
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
path_checkpoint_relation_network = dict_conf['checkpoints']['path_relation_network']
path_checkpoint_att_module = dict_conf['checkpoints']['path_att_module']
distance = dict_conf['distance']
pooling = conf_tsm['pooling']

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

def test(feature_encoder, transform_test, meta_data, relation_network, episodes, att_module, taskGen, print_enable = False):
    print("Testing... N_episode: {}".format(episodes), flush=True)
    accuracies = []

    feature_encoder.eval()

    task_val = taskGen(meta_data, CLASS_NUM, SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS, path_frames, num_snippets = conf_tsm['num_segments'], num_episodes = episodes)
    sample_dataloader_val = tg.get_data_loader(task_val,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False, transforms = transform_test,  test_mode = True, workers = pool)
    batch_dataloader_val = tg.get_data_loader(task_val,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True, transforms = transform_test,  test_mode = True, workers = pool)
    it_sampleloader_val = iter(sample_dataloader_val)
    it_batchloader_val = iter(batch_dataloader_val)
    
    for i in range(episodes):
        
        sample_images,sample_labels = next(it_sampleloader_val)
        sample_images,sample_labels = torch.squeeze(sample_images, dim =0),torch.squeeze(sample_labels, dim = 0)
        batches,batch_labels = next(it_batchloader_val)
        batches,batch_labels = torch.squeeze(batches, dim =0),torch.squeeze(batch_labels, dim = 0)

        sample_features = feature_encoder(sample_images.to(device)) # 25*64*19*19
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,sample_features.size(1),sample_features.size(2), sample_features.size(3), sample_features.size(4))
        sample_features = torch.sum(sample_features,1).squeeze(1)

        batch_features = feature_encoder(batches.to(device))

        # calculate relations
        # each batch sample link to every samples to calculate relations    
        
        if distance == 'relationalNetwork':
            sample_features = sample_features.permute(0, 2, 1, 3, 4)
            batch_features = batch_features.permute(0, 2, 1, 3, 4)

            pool_sample_features = torch.unsqueeze(CovpoolLayer(sample_features), dim = 1)
            pool_batch_features = torch.unsqueeze(CovpoolLayer(batch_features), dim = 1)

            size_sample = pool_sample_features.size()
            pool_sample_features = torch.unsqueeze(pool_sample_features, dim = 0).expand(BATCH_NUM_PER_CLASS*CLASS_NUM, size_sample[0], size_sample[1], size_sample[2], size_sample[3])
            
            size_batch = pool_batch_features.size()
            pool_batch_features = torch.unsqueeze(pool_batch_features, dim = 1).expand(size_batch[0], CLASS_NUM, size_batch[1], size_batch[2], size_batch[3])
            
            relation_pairs = torch.cat((pool_sample_features, pool_batch_features), 2).view(-1,1*2,64,64)
            relation_pairs = PowerNorm(relation_pairs)
            relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
        else:
            if pooling == 'second_order':
                sample_features = sample_features.permute(0, 2, 1, 3, 4)
                batch_features = batch_features.permute(0, 2, 1, 3, 4)
            
                pool_sample_features = torch.unsqueeze(CovpoolLayer(sample_features), dim = 1)
                pool_batch_features = torch.unsqueeze(CovpoolLayer(batch_features), dim = 1)
            elif pooling == 'attention':
                sample_features = sample_features.view(sample_features.size(0), sample_features.size(1), -1)
                att_sample_features = att_module(sample_features)

                batch_features = batch_features.view(batch_features.size(0), batch_features.size(1), -1)
                att_batch_features = att_module(batch_features)

                att_sample_features = att_sample_features.expand_as(sample_features)
                pool_sample_features = sample_features*att_sample_features
                pool_sample_features = torch.sum(pool_sample_features, dim = 1)

                att_batch_features = att_batch_features.expand_as(batch_features)
                pool_batch_features = batch_features*att_batch_features
                pool_batch_features = torch.sum(pool_batch_features, dim = 1)
            else:
                pool_sample_features = torch.mean(sample_features, dim = 1)
                pool_batch_features = torch.mean(batch_features, dim = 1)

            size_sample = pool_sample_features.size()
            pool_sample_features = pool_sample_features.view(size_sample[0], -1)
            size_sample = pool_sample_features.size()
            pool_sample_features = torch.unsqueeze(pool_sample_features, dim = 0).expand(BATCH_NUM_PER_CLASS*CLASS_NUM, size_sample[0], size_sample[1])

            size_batch = pool_batch_features.size()
            pool_batch_features = pool_batch_features.view(size_batch[0], -1)
            size_batch = pool_batch_features.size()
            pool_batch_features = torch.unsqueeze(pool_batch_features, dim = 1).expand(size_batch[0], CLASS_NUM, size_batch[1])

            relations = relation_network(pool_sample_features, pool_batch_features)

        batch_labels = batch_labels.type(torch.LongTensor)

        accuracy = count_accuracy(relations.data, batch_labels)
        accuracies.append(accuracy)

    mean_accuracy, h = mean_confidence_interval(accuracies)

    return mean_accuracy, h

def train(type_loss, feature_encoder, feature_encoder_optim, feature_encoder_scheduler, transform_train, transform_test, relation_network, relation_network_optim, relation_network_scheduler, att_module, att_module_optim, att_module_scheduler, taskGen, metatrain_data, metaval_data):

    print("Training...", flush=True)

    last_accuracy = 0

    if type_loss == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss().to(device)
    else:
        loss_fn = nn.MSELoss().to(device)

    with experiment.train():

        task = taskGen(metatrain_data, CLASS_NUM, SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS, path_frames, num_snippets = conf_tsm['num_segments'], num_episodes = EPISODE)
    
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False, transforms = transform_train,  test_mode = False, workers = pool)
        batch_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True, transforms = transform_train,  test_mode = False, workers = pool)
        it_sampleloader = iter(sample_dataloader)
        it_batchloader = iter(batch_dataloader)

        for episode in range(EPISODE):

            if conf_tsm['no_partialbn']:
                if torch.cuda.device_count() > 1:
                    feature_encoder.module.TSM.partialBN(False)
                else:
                    feature_encoder.TSM.partialBN(False)
            else:
                if torch.cuda.device_count() > 1:
                    feature_encoder.module.TSM.partialBN(True)
                else:
                    feature_encoder.TSM.partialBN(True)
            
            feature_encoder.train()

            # adjust_learning_rate(feature_encoder_optim, epoch, conf_tsm['lr_type'], conf_tsm['lr_steps'])

            # init dataset
            # sample_dataloader is to obtain previous samples for compare
            # batch_dataloader is to batch samples for training
            
            # sample datas
            samples,sample_labels = next(it_sampleloader) #25*3*84*84
            samples,sample_labels = torch.squeeze(samples, dim =0),torch.squeeze(sample_labels, dim = 0)
            batches,batch_labels = next(it_batchloader)
            batches,batch_labels = torch.squeeze(batches, dim =0),torch.squeeze(batch_labels, dim = 0)
        
            batch_labels = batch_labels.type(torch.LongTensor)
            samples = samples.to(device)
            batches = batches.to(device)
            # calculate features
            sample_features = feature_encoder(samples) # 25*64*19*19
            sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,sample_features.size(1),sample_features.size(2), sample_features.size(3), sample_features.size(4))
            sample_features = torch.sum(sample_features,1).squeeze(1)
            batch_features = feature_encoder(batches) # 20x64*5*5

            # calculate relations
            # each batch sample link to every samples to calculate relations
            

            if distance == 'relationalNetwork':
                sample_features = sample_features.permute(0, 2, 1, 3, 4)
                batch_features = batch_features.permute(0, 2, 1, 3, 4)

                pool_sample_features = torch.unsqueeze(CovpoolLayer(sample_features), dim = 1)
                pool_batch_features = torch.unsqueeze(CovpoolLayer(batch_features), dim = 1)

                size_sample = pool_sample_features.size()
                pool_sample_features = torch.unsqueeze(pool_sample_features, dim = 0).expand(BATCH_NUM_PER_CLASS*CLASS_NUM, size_sample[0], size_sample[1], size_sample[2], size_sample[3])
                # print(pool_sample_features.size())

                size_batch = pool_batch_features.size()
                pool_batch_features = torch.unsqueeze(pool_batch_features, dim = 1).expand(size_batch[0], CLASS_NUM, size_batch[1], size_batch[2], size_batch[3])
                # print(pool_batch_features.size())
                
                relation_pairs = torch.cat((pool_sample_features, pool_batch_features), 2).view(-1,1*2,64,64)
                relation_pairs = PowerNorm(relation_pairs)
                relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
            else:
                if pooling == 'second_order':
                    sample_features = sample_features.permute(0, 2, 1, 3, 4)
                    batch_features = batch_features.permute(0, 2, 1, 3, 4)

                    pool_sample_features = torch.unsqueeze(CovpoolLayer(sample_features), dim = 1)
                    pool_batch_features = torch.unsqueeze(CovpoolLayer(batch_features), dim = 1)
                elif pooling == 'attention':
                    sample_features = sample_features.view(sample_features.size(0), sample_features.size(1), -1)
                    att_sample_features = att_module(sample_features)

                    batch_features = batch_features.view(batch_features.size(0), batch_features.size(1), -1)
                    att_batch_features = att_module(batch_features)

                    att_sample_features = att_sample_features.expand_as(sample_features)
                    pool_sample_features = sample_features*att_sample_features
                    pool_sample_features = torch.sum(pool_sample_features, dim = 1)

                    att_batch_features = att_batch_features.expand_as(batch_features)
                    pool_batch_features = batch_features*att_batch_features
                    pool_batch_features = torch.sum(pool_batch_features, dim = 1)
                else:
                    # print('sample_features: ',sample_features.size())
                    pool_sample_features = torch.mean(sample_features, dim = 1)
                    pool_batch_features = torch.mean(batch_features, dim = 1)

                size_sample = pool_sample_features.size()
                pool_sample_features = pool_sample_features.view(size_sample[0], -1)
                size_sample = pool_sample_features.size()
                pool_sample_features = torch.unsqueeze(pool_sample_features, dim = 0).expand(BATCH_NUM_PER_CLASS*CLASS_NUM, size_sample[0], size_sample[1])

                size_batch = pool_batch_features.size()
                pool_batch_features = pool_batch_features.view(size_batch[0], -1)
                size_batch = pool_batch_features.size()
                pool_batch_features = torch.unsqueeze(pool_batch_features, dim = 1).expand(size_batch[0], CLASS_NUM, size_batch[1])

                relations = relation_network(pool_sample_features, pool_batch_features)

            if type_loss =='MSELoss': 
                one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM, dtype=torch.long).scatter_(1, batch_labels.view(-1,1), 1).to(device))
            else:
                one_hot_labels = batch_labels.to(device)

            loss = loss_fn(relations,one_hot_labels)
            acc = count_accuracy(relations, batch_labels)
            # training

            feature_encoder.zero_grad()
            if distance == 'relationalNetwork':
                relation_network.zero_grad()

            if pooling == 'attention':
                att_module.zero_grad()

            loss.backward()

            if conf_tsm['clip_gradient'] is not None:
                torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), conf_tsm['clip_gradient'])
                if pooling == 'attention':
                    torch.nn.utils.clip_grad_norm_(att_module.parameters(), conf_tsm['clip_gradient'])

            #torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
            if distance == 'relationalNetwork':
                torch.nn.utils.clip_grad_norm_(relation_network.parameters(),0.5)

            feature_encoder_optim.step()
            if distance == 'relationalNetwork':
                relation_network_optim.step()
            
            if pooling == 'attention':
                att_module_optim.step()

            feature_encoder_scheduler.step()
            if distance == 'relationalNetwork':
                relation_network_scheduler.step()

            if pooling == 'attention':
                att_module_scheduler.step()

            experiment.log_metric("accuracy", acc, step=episode)
            experiment.log_metric("loss", loss.item(), step=episode)

            if (episode)%num_episode_to_val == 0:
                # test
                with experiment.validate():
                    test_accuracy, h = test(feature_encoder, transform_test, metaval_data, relation_network, VAL_EPISODE, att_module, taskGen, False)
                    print("test Accuracy: {:.2f} ± {:.2f} %".format(test_accuracy,h), flush=True)
                    experiment.log_metric("accuracy", test_accuracy, step = episode)
                    experiment.log_metric("confidence interval", h, step = episode)

                    if test_accuracy > last_accuracy:
                        # save networks
                        print('Saving ... ')
                        torch.save(feature_encoder.state_dict(),str(path_checkpoint_feature_encoder + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                        if distance == 'relationalNetwork':
                            torch.save(relation_network.state_dict(),str(path_checkpoint_relation_network + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                        if pooling == 'attention':
                            torch.save(att_module.state_dict(),str(path_checkpoint_att_module + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

                        print("save networks for episode:",episode, flush=True)

                        last_accuracy = test_accuracy



def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction

    # Step 2: init neural networks
    print("init neural networks", flush=True)

    feature_encoder = CNNEncoder(conf_tsm)
    if distance == 'relationalNetwork':
        relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)
        print('Using Relational Network')
    else:
        relation_network = nn.CosineSimilarity(dim=2, eps=1e-6)
        print('Using Cosine Distance')

    # input_size = feature_encoder.TSM.input_size
    crop_size = feature_encoder.TSM.crop_size
    scale_size = feature_encoder.TSM.scale_size
    input_mean = feature_encoder.TSM.input_mean
    input_std = feature_encoder.TSM.input_std
    policies = feature_encoder.TSM.get_optim_policies()

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

    train_augmentation = feature_encoder.TSM.get_augmentation(flip=False if 'something' in dataset_name or 'jester' in dataset_name else True)
    taskGen = tg.S2Sv2Task if 'something' in dataset_name else tg.KineticsTask


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

    #Agregar test Trans

    #feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    if torch.cuda.device_count() > 1:
        feature_encoder = nn.DataParallel(feature_encoder)
        if distance == 'relationalNetwork':
            relation_network = nn.DataParallel(relation_network)
    
    print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)

    feature_encoder.to(device)
    if distance == 'relationalNetwork':
        relation_network.to(device)

    if conf_tsm['pooling'] == 'attention':
        t_dim = conf_tsm['num_segments']
        radio = conf_tsm['ratio']
        chann_dim = conf_tsm['chann_dim']
        att_module = AttModule(radio, t_dim, chann_dim)
        att_module.to(device)

        att_module_optim = torch.optim.SGD(att_module.parameters(), conf_tsm['lr'], momentum=conf_tsm['momentum'], weight_decay=conf_tsm['weight_decay'])
        att_module_scheduler = StepLR(att_module_optim,step_size=num_episode_decay,gamma=0.1)
    else:
        att_module = None
        att_module_optim = None
        att_module_scheduler = None
    
    feature_encoder_optim = torch.optim.SGD(policies, conf_tsm['lr'], momentum=conf_tsm['momentum'], weight_decay=conf_tsm['weight_decay'])
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=num_episode_decay,gamma=0.1)

    if distance == 'relationalNetwork':
        relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=conf_tsm['lr'])
        relation_network_scheduler = StepLR(relation_network_optim,step_size=num_episode_decay,gamma=0.1)
        type_loss = 'MSELoss'
    else: 
        relation_network_optim = None
        relation_network_scheduler = None
        type_loss = 'CrossEntropyLoss'


    if os.path.exists(str(path_checkpoint_feature_encoder + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str(path_checkpoint_feature_encoder + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str(path_checkpoint_relation_network + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")) and distance == 'relationalNetwork':
        relation_network.load_state_dict(torch.load(str(path_checkpoint_relation_network + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")
    if os.path.exists(str(path_checkpoint_att_module + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")) and pooling == 'attention':
        att_module.load_state_dict(torch.load(str(path_checkpoint_att_module + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load attention network success")

    if dict_conf['train_mode']:
        metatrain_data, num_vid_train = tg.get_data_classes(path_metatrain, path_frames, conf_tsm['num_segments'], SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        metaval_data, num_vid_val = tg.get_data_classes(path_metaval, path_frames, conf_tsm['num_segments'], SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        print("num_vid_train: {}, num_vid_val: {}".format(num_vid_train, num_vid_val))
        train(type_loss, feature_encoder, feature_encoder_optim, feature_encoder_scheduler, transform_train, transform_test, relation_network, relation_network_optim, relation_network_scheduler, att_module, att_module_optim, att_module_scheduler, taskGen, metatrain_data, metaval_data)
        if os.path.exists(str(path_checkpoint_feature_encoder + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
            feature_encoder.load_state_dict(torch.load(str(path_checkpoint_feature_encoder + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
            print("load feature encoder success")
        if os.path.exists(str(path_checkpoint_relation_network + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")) and distance == 'relationalNetwork':
            relation_network.load_state_dict(torch.load(str(path_checkpoint_relation_network + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
            print("load relation network success")
        if os.path.exists(str(path_checkpoint_att_module + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")) and pooling == 'attention':
            att_module.load_state_dict(torch.load(str(path_checkpoint_att_module + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
            print("load attention network success")
    
    with experiment.test():
        print("Init test")
        metatest_data, num_vid_test = tg.get_data_classes(path_metatest, path_frames, conf_tsm['num_segments'], SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        mean_accuracy, h = test(feature_encoder, transform_test, metatest_data, relation_network, TEST_EPISODE, att_module, taskGen, True)
        print("test Accuracy: {:.2f} ± {:.2f} %".format(mean_accuracy,h), flush=True)
        experiment.log_metric("accuracy", mean_accuracy)
        experiment.log_metric("confidence interval", h)
                




if __name__ == '__main__':
    main()
