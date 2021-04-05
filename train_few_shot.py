
from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import task_generator as tg
import os
import math
import argparse
import scipy as sp
import scipy.stats
from model.rnact import CNNEncoder,RelationNetwork,CovpoolLayer,PowerNorm
import statistics 
import yaml

parser = argparse.ArgumentParser(description="Few Shot Video Recognition")
parser.add_argument("-conf","--conf_path", default = './conf/conf_base.yaml')
args = parser.parse_args()
conf_file = open(args.conf_path, 'r')
dict_conf = yaml.load(conf_file)

# parser.add_argument("-f","--feature_dim",type = int, default = 1024)
# parser.add_argument("-r","--relation_dim",type = int, default = 16)
# parser.add_argument("-w","--class_num",type = int, default = 5)
# parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
# parser.add_argument("-b","--batch_num_per_class",type = int, default = 10)
# parser.add_argument("-e","--episode",type = int, default= 10000)
# parser.add_argument("-t","--test_episode", type = int, default = 600)
# parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
# parser.add_argument("-g","--gpu",type=int, default=0)
# parser.add_argument("-key","--api_key", type=String, default='FbkM3YZFUZNIIUQ4EDJc8wsBv')
# args = parser.parse_args()


# Hyper Parameters
FEATURE_DIM = dict_conf['feature_dim']
RELATION_DIM = dict_conf['relation_dim']
CLASS_NUM = dict_conf['class_num']
SAMPLE_NUM_PER_CLASS = dict_conf['sample_num_per_class']
BATCH_NUM_PER_CLASS = dict_conf['batch_num_per_class']
EPISODE = dict_conf['episode']
TEST_EPISODE = dict_conf['test_episode']
LEARNING_RATE = dict_conf['learning_rate']
api_key = dict_conf['comet']['api_key']
workspace = dict_conf['comet']['workspace']
project_name = dict_conf['comet']['project_name']

experiment = Experiment(api_key=api_key,
                        project_name=project_name, workspace=workspace)
experiment.log_parameters(dict_conf)

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

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    
    path_metatrain = dict_conf['dataset']['path_metatrain']
    path_metaval = dict_conf['dataset']['path_metaval']
    path_metatest = dict_conf['dataset']['path_metatest']
    path_frames = dict_conf['dataset']['path_frames']
    path_checkpoint_feature_encoder = dict_conf['checkpoints']['path_feature_encoder']
    path_checkpoint_relation_network = dict_conf['checkpoints']['path_relation_network']

    # Step 2: init neural networks
    print("init neural networks", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM,RELATION_DIM)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    if torch.cuda.device_count() > 1:
        feature_encoder = nn.DataParallel(feature_encoder)
        relation_network = nn.DataParallel(relation_network)
    
    print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)

    feature_encoder.to(device)
    relation_network.to(device)

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(),lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    if os.path.exists(str(path_checkpoint_feature_encoder + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str(path_checkpoint_feature_encoder + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str(path_checkpoint_relation_network + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str(path_checkpoint_relation_network + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")

    # Step 3: build graph
    print("Training...", flush=True)

    last_accuracy = 0.0

    mse = nn.MSELoss().to(device)

    with experiment.train():

        for episode in range(EPISODE):

            feature_encoder_scheduler.step(episode)
            relation_network_scheduler.step(episode)

            # init dataset
            # sample_dataloader is to obtain previous samples for compare
            # batch_dataloader is to batch samples for training
            task = tg.KineticsTask(path_metatrain, CLASS_NUM, SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS, path_frames, num_snippets = 20)
    
            sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
            batch_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True)
            
            # sample datas
            samples,sample_labels = sample_dataloader.__iter__().next() #25*3*84*84
            batches,batch_labels = batch_dataloader.__iter__().next()
            
            batch_labels = batch_labels.type(torch.LongTensor)
            
            # calculate features
            sample_features = feature_encoder(samples.to(device)) # 25*64*19*19
            sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,sample_features.size(1),sample_features.size(2), sample_features.size(3), sample_features.size(4))
            sample_features = torch.sum(sample_features,1).squeeze(1)
            batch_features = feature_encoder(batches.to(device)) # 20x64*5*5

            # calculate relations
            # each batch sample link to every samples to calculate relations
            pool_sample_features = torch.unsqueeze(CovpoolLayer(sample_features), dim = 1)
            pool_batch_features = torch.unsqueeze(CovpoolLayer(batch_features), dim = 1)
            
            size_sample = pool_sample_features.size()
            pool_sample_features = torch.unsqueeze(pool_sample_features, dim = 0).expand(BATCH_NUM_PER_CLASS*CLASS_NUM, size_sample[0], size_sample[1], size_sample[2], size_sample[3])
            
            size_batch = pool_batch_features.size()
            pool_batch_features = torch.unsqueeze(pool_batch_features, dim = 1).expand(size_batch[0], CLASS_NUM, size_batch[1], size_batch[2], size_batch[3])
            
            relation_pairs = torch.cat((pool_sample_features, pool_batch_features), 2).view(-1,1*2,64,64)
            relation_pairs = PowerNorm(relation_pairs)
            relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
            
            # to form a 100x128 matrix for relation network
    #         sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
    #         batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
    #         batch_features_ext = torch.transpose(batch_features_ext,0,1)
    #         relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,FEATURE_DIM*2,19,19)
    #         relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
            
            one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1).to(device))
            loss = mse(relations,one_hot_labels)

            _,ind_rel = torch.max(relations, 1)
            _,ind_true = torch.max(one_hot_labels, 1)
            acc = torch.sum((ind_rel == ind_true)).item()/ind_rel.size(0)
            # training

            feature_encoder.zero_grad()
            relation_network.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm(feature_encoder.parameters(),0.5)
            torch.nn.utils.clip_grad_norm(relation_network.parameters(),0.5)

            feature_encoder_optim.step()
            relation_network_optim.step()

            experiment.log_metric("accuracy", acc, step=episode)
            experiment.log_metric("loss", loss.item(), step=episode)

            if (episode+1)%100 == 0:
                    print("episode:",episode+1,"loss",loss.item())

            if (episode)%100 == 0:

                # test
                with experiment.test():
                    print("Testing... N_episode: {}".format(TEST_EPISODE), flush=True)
                    accuracies = []
                    for i in range(TEST_EPISODE):
                        task = tg.KineticsTask(path_metatest, CLASS_NUM, SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS, path_frames, num_snippets = 20)
                        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False)
                        batch_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="test",shuffle=True)

                        sample_images,sample_labels = sample_dataloader.__iter__().next()
                        batches,batch_labels = batch_dataloader.__iter__().next()

                        sample_features = feature_encoder(sample_images.to(device)) # 25*64*19*19
                        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,sample_features.size(1),sample_features.size(2), sample_features.size(3), sample_features.size(4))
                        sample_features = torch.sum(sample_features,1).squeeze(1)

                        batch_features = feature_encoder(batches.to(device))

                        # calculate relations
                        # each batch sample link to every samples to calculate relations
                        pool_sample_features = torch.unsqueeze(CovpoolLayer(sample_features), dim = 1)
                        pool_batch_features = torch.unsqueeze(CovpoolLayer(batch_features), dim = 1)
                        
                        size_sample = pool_sample_features.size()
                        pool_sample_features = torch.unsqueeze(pool_sample_features, dim = 0).expand(SAMPLE_NUM_PER_CLASS*CLASS_NUM, size_sample[0], size_sample[1], size_sample[2], size_sample[3])
                        
                        size_batch = pool_batch_features.size()
                        pool_batch_features = torch.unsqueeze(pool_batch_features, dim = 1).expand(size_batch[0], CLASS_NUM, size_batch[1], size_batch[2], size_batch[3])
                        
                        relation_pairs = torch.cat((pool_sample_features, pool_batch_features), 2).view(-1,1*2,64,64)
                        relation_pairs = PowerNorm(relation_pairs)
                        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)
                        _,predict_labels = torch.max(relations.data,1)

                        predict_labels = predict_labels.cpu()
                        batch_labels = batch_labels.type(torch.LongTensor)

                        rewards = [1 if predict_labels[j]==batch_labels[j] else 0 for j in range(CLASS_NUM*SAMPLE_NUM_PER_CLASS)]

                        accuracy = np.sum(rewards)/len(rewards)
                        accuracies.append(accuracy)
                    test_accuracy = statistics.mean(accuracies)
                    std_accuracy = statistics.stdev(accuracies)
                    print("test accuracy: {}, std accuracy: {}".format(test_accuracy, std_accuracy), flush=True)
                    experiment.log_metric("accuracy", test_accuracy, step = episode)
                    experiment.log_metric("std_accuracy", std_accuracy, step = episode)

                    if test_accuracy > last_accuracy:

                        # save networks
                        torch.save(feature_encoder.state_dict(),str(path_checkpoint_feature_encoder + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                        torch.save(relation_network.state_dict(),str(path_checkpoint_relation_network + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

                        print("save networks for episode:",episode, flush=True)

                        last_accuracy = test_accuracy


                #     for test_images,test_labels in batch_features:
                #         batch_size = test_labels.shape[0]
                        
                #         # calculate features
                #         sample_features = feature_encoder(sample_images.cuda(GPU)) # 25*64*19*19
                #         sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,sample_features.size(1),sample_features.size(2), sample_features.size(3), sample_features.size(4))
                #         sample_features = torch.sum(sample_features,1).squeeze(1)
                #         test_features = feature_encoder(test_images.cuda(GPU)) # 20x64*5*5
                        
                #         pool_sample_features = torch.unsqueeze(CovpoolLayer(sample_features), dim = 1)
                #         pool_test_features = torch.unsqueeze(CovpoolLayer(test_features), dim = 1)
                        
                        
                #         size_sample = pool_sample_features.size()
                #         pool_sample_features = torch.unsqueeze(pool_sample_features, dim = 0).expand(batch_size, size_sample[0], size_sample[1], size_sample[2], size_sample[3])

                #         size_test = pool_test_features.size()
                #         pool_test_features = torch.unsqueeze(pool_test_features, dim = 1).expand(size_test[0], CLASS_NUM, size_test[1], size_test[2], size_test[3])

                #         relation_pairs = torch.cat((pool_sample_features, pool_test_features), 2).view(-1,1*2,64,64)
                #         relation_pairs = PowerNorm(relation_pairs)
                #         relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

                #         _,predict_labels = torch.max(relations.data,1)
                        
                #         predict_labels = predict_labels.cpu()
                #         test_labels = test_labels.type(torch.LongTensor)

                #         rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(batch_size)]

                #         total_rewards += np.sum(rewards)


                #     accuracy = total_rewards/1.0/CLASS_NUM/15
                #     accuracies.append(accuracy)


                # test_accuracy,h = mean_confidence_interval(accuracies)

                # print("test accuracy:",test_accuracy,"h:",h, flush=True)

                




if __name__ == '__main__':
    main()
