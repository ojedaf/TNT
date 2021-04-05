import torch
import torch.nn as nn
import torch.nn.functional as F
from .TRNModule import return_TRN


class ClassRepresentationEncoder(nn.Module):
    """
    Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
    on sets (mainly for extracting task-level representations from context sets).
    """
    def __init__(self, model_dim, task_dim, type_repre = 'Aug', num_examples = 5, epsilon = 10):
        super(ClassRepresentationEncoder, self).__init__()

        print("Added Class Representation Encoder")
        print("Model Dim: {}, Task Dim: {}".format(model_dim, task_dim))

        self.type_repre = type_repre
        if self.type_repre == 'Aug':
            print('Aug mode')
            self.linear1 = nn.Sequential(
                nn.Linear(768, model_dim),
                nn.ReLU()
            )

            self.cos_dis = nn.CosineSimilarity(dim=2, eps=1e-6)
            self.activation = nn.Softmax(dim=1)
            self.epsilon = epsilon
            self.num_examples = num_examples
        
        else:
            print('context mode')
            new_dim = model_dim+task_dim
            self.norm = nn.LayerNorm(new_dim)
            self.linear1 = nn.Sequential(
                nn.Linear(new_dim, model_dim),
                nn.ReLU()
            )  
        

    def forward(self, support_features, task_embeddings):

        if self.type_repre == 'Aug':
            task_embeddings = self.linear1(task_embeddings)
            task_embeddings = task_embeddings.view(-1, self.num_examples, task_embeddings.size(1))
            support_features = support_features.view(-1, self.num_examples, support_features.size(1))

            augmented_support_features = torch.cat((support_features, task_embeddings), dim = 1)

            P = torch.mean(support_features, dim = 1, keepdim=True)
            
            P = P.expand_as(augmented_support_features)
            
            weights = self.activation(self.epsilon*self.cos_dis(P, augmented_support_features))
            
            weights = weights.unsqueeze(2).expand_as(augmented_support_features)

            augmented_support_features = augmented_support_features*weights

            return augmented_support_features.view(-1, augmented_support_features.size(2))
        
        else:

            support_features = torch.cat((support_features, task_embeddings), dim = 1)
            support_features = self.norm(support_features)
            support_features = self.linear1(support_features)

            return support_features


class AttentionBasedRepresentationEncoder(nn.Module):
    """
    Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
    on sets (mainly for extracting task-level representations from context sets).
    """
    def __init__(self, model_dim, task_dim, num_heads, temp_pooling, num_frames, num_examples = 5):
        super(AttentionBasedRepresentationEncoder, self).__init__()

        print("Added Class Attention based Representation Encoder")
        print("Model Dim: {}, Task Dim: {}, num_heads: {}, num_examples: {}".format(model_dim, task_dim, num_heads, num_examples))

        self.multihead_attn = nn.MultiheadAttention(embed_dim = model_dim, num_heads = num_heads)
        self.num_examples = num_examples
        self.num_frames = num_frames
        self.temp_pooling = temp_pooling
        if temp_pooling in ['TRN', 'TRNmultiscale']:
            print('temp pooling: ',temp_pooling)
            self.img_feature_dim = 256
            self.new_fc = nn.Linear(model_dim, self.img_feature_dim)
            self.temp_consensus = return_TRN(relation_type = temp_pooling, img_feature_dim = self.img_feature_dim, num_frames = num_frames, out_dim = self.img_feature_dim)
            self.conv2d = nn.Conv2d(in_channels=self.img_feature_dim, out_channels=task_dim, kernel_size = 1)
        else: 
            print('temp pooling: mean')
            self.temp_consensus = self.mean_pooling
            self.conv2d = nn.Conv2d(in_channels=model_dim, out_channels=task_dim, kernel_size = 1)
    
    def mean_pooling(self, x):
        return torch.mean(x, dim=1)

    def forward(self, query_features, support_features, task_embeddings):

        num_frames = query_features.size(1)
        
        assert num_frames == self.num_frames and len(query_features.size()) == 3 and len(support_features.size()) == 3, "The tensors must be R3 and have {} frames".format(self.num_frames)
        
        if self.temp_pooling in ['TRN', 'TRNmultiscale']:

            query_features = query_features.view(-1, query_features.size(2))
            support_features = support_features.view(-1, support_features.size(2))

            query_features = self.new_fc(query_features)
            support_features = self.new_fc(support_features)

            query_features = query_features.view(-1, num_frames, query_features.size(1))
            support_features = support_features.view(-1, num_frames, support_features.size(1))

            query_features = self.temp_consensus(query_features)
            support_features = self.temp_consensus(support_features)

        else:
            
            query_features = self.temp_consensus(query_features)
            support_features = self.temp_consensus(support_features)

        # query_features = query_features.unsqueeze(2).unsqueeze(3)
        # support_features = support_features.unsqueeze(2).unsqueeze(3)

        # support_features = self.conv2d(support_features)
        # query_features = self.conv2d(query_features)

        # support_features = support_features.squeeze()
        # query_features = query_features.squeeze()

        distribution_memory = torch.cat((query_features, support_features), dim = 0)

        task_embeddings = task_embeddings.view(-1, self.num_examples, task_embeddings.size(1))
        task_embeddings = torch.mean(task_embeddings, dim = 1, keepdim=True)

        memory_unlabeled = torch.unsqueeze(query_features, dim = 1)

        _, att_wts = self.multihead_attn(task_embeddings, memory_unlabeled, memory_unlabeled)
        att_wts = att_wts[0]
        n_class = att_wts.size(0)
        n_exampsup = self.num_examples

        att_sup = torch.zeros(n_class, n_class*n_exampsup).to(att_wts.device)
        for i in range(n_class):
            att_sup[i, n_exampsup*i:n_exampsup*(i+1)] = 1/n_exampsup

        att_wts = torch.cat((att_wts, att_sup), dim = 1)

        ws = att_wts.unsqueeze(2).expand(att_wts.size(0), att_wts.size(1), distribution_memory.size(1))
        distribution_memory = distribution_memory.unsqueeze(0).expand(att_wts.size(0), distribution_memory.size(0), distribution_memory.size(1))

        class_prototipies = torch.sum(ws*distribution_memory, dim = 1)
        Ntw = torch.sum(att_wts, dim = 1)
        Ntw = Ntw.unsqueeze(1).expand_as(class_prototipies)

        class_prototipies = class_prototipies*(1/Ntw)

        return class_prototipies, support_features, query_features

class LabelPropagation(nn.Module):
    def __init__(self, num_frames, num_examples = 5):
        super(LabelPropagation, self).__init__()
        print('Label Propagation Module was added')
        print('temp pooling: mean')
        self.temp_consensus = self.mean_pooling
        self.num_frames = num_frames
        self.num_examples = num_examples
        self.relation_module = nn.CosineSimilarity(dim=2, eps=1e-6)
    
    def mean_pooling(self, x):
        return torch.mean(x, dim=1)

    def forward(self, query_features, support_features, task_embeddings = None):

        num_frames = query_features.size(1)
        
        assert num_frames == self.num_frames and len(query_features.size()) == 3 and len(support_features.size()) == 3, "The tensors must be R3 and have {} frames".format(self.num_frames)

        query_features = self.temp_consensus(query_features)
        support_features = self.temp_consensus(support_features)

        init_proto = support_features.view(-1, self.num_examples, support_features.size(1))
        init_proto = torch.mean(init_proto, dim = 1)

        size_sample = init_proto.size()
        size_batch = query_features.size()
        init_proto = torch.unsqueeze(init_proto, dim = 1).expand(size_sample[0], size_batch[0], size_sample[1])

        instance_memory = torch.unsqueeze(query_features, dim = 0).expand(size_sample[0], size_batch[0], size_batch[1])

        pre_relations = self.relation_module(init_proto, instance_memory)
        att_wts = F.softmax(pre_relations, 1)

        att_sup = torch.zeros(size_sample[0], size_sample[0]*self.num_examples).to(att_wts.device)
        for i in range(size_sample[0]):
            att_sup[i, self.num_examples*i:self.num_examples*(i+1)] = 1/self.num_examples

        att_wts = torch.cat((att_wts, att_sup), dim = 1)
        all_instances = torch.cat((query_features, support_features), 0)

        ws = att_wts.unsqueeze(2).expand(att_wts.size(0), att_wts.size(1), all_instances.size(1))
        all_instances = all_instances.unsqueeze(0).expand(att_wts.size(0), all_instances.size(0), all_instances.size(1))

        class_proto = torch.sum(ws*all_instances, dim = 1)
        Ntw = torch.sum(att_wts, dim = 1)
        Ntw = Ntw.unsqueeze(1).expand_as(class_proto)

        class_proto = class_proto*(1/Ntw)

        return class_proto, support_features, query_features

class TwoLevelAttentionBasedRepresentationEncoder(nn.Module):
    """
    Simple set encoder, implementing the DeepSets approach. Used for modeling permutation invariant representations
    on sets (mainly for extracting task-level representations from context sets).
    """
    def __init__(self, model_dim, task_dim, num_heads, num_examples = 5):
        super(TwoLevelAttentionBasedRepresentationEncoder, self).__init__()

        print("Added Two Level Class Attention based Representation Encoder")
        print("Model Dim: {}, Task Dim: {}, num_heads: {}, num_examples: {}".format(model_dim, task_dim, num_heads, num_examples))

        self.conv2d = nn.Conv2d(in_channels=model_dim, out_channels=task_dim, kernel_size = 1)
        self.multihead_attn_temp_level = nn.MultiheadAttention(embed_dim = task_dim, num_heads = num_heads)
        self.multihead_attn_class_level = nn.MultiheadAttention(embed_dim = task_dim, num_heads = num_heads)
        self.multihead_attn_video_mem = nn.MultiheadAttention(embed_dim = task_dim, num_heads = num_heads)
        self.num_examples = num_examples

    def forward(self, query_features, support_features, task_embeddings):
        
        num_frames = query_features.size(1)
        query_features = query_features.view(-1, query_features.size(2))
        support_features = support_features.view(-1, support_features.size(2))

        query_features = query_features.unsqueeze(2).unsqueeze(3)
        support_features = support_features.unsqueeze(2).unsqueeze(3)

        support_features = self.conv2d(support_features)
        query_features = self.conv2d(query_features)

        support_features = support_features.squeeze()
        query_features = query_features.squeeze()

        query_features = query_features.view(-1, num_frames, query_features.size(1))
        support_features = support_features.view(-1, num_frames, support_features.size(1))

        distribution_memory = torch.cat((query_features, support_features), dim = 0) # (K(N+M), T, D)
        distribution_memory = distribution_memory.permute(1, 0, 2)

        task_embeddings = task_embeddings.view(-1, self.num_examples, task_embeddings.size(1))
        task_embeddings = torch.mean(task_embeddings, dim = 1, keepdim=True)

        class_prototipies = []
        class_memories = []
        for i in range(task_embeddings.size(0)):
            class_emb = task_embeddings[i,:,:]
            class_emb_temp = class_emb.expand(distribution_memory.size(1), class_emb.size(1)).unsqueeze(0)
            
            memory_class = self.multihead_attn_temp_level(class_emb_temp, distribution_memory, distribution_memory)[0]
            memory_class = memory_class.view(-1, memory_class.size(2)).unsqueeze(1)
            class_emb = class_emb.unsqueeze(0)
            class_proto = self.multihead_attn_class_level(class_emb, memory_class, memory_class)[0]

            class_prototipies.append(class_proto)
            memory_class = torch.squeeze(memory_class)
            class_memories.append(memory_class)
        
        class_prototipies = torch.stack(class_prototipies, dim = 0)
        class_prototipies = torch.squeeze(class_prototipies)

        class_memories = torch.stack(class_memories, dim = 0)
        distribution_memory = torch.mean(distribution_memory, dim = 0, keepdim = True)
        
        distribution_memory = self.multihead_attn_video_mem(distribution_memory, class_memories, class_memories)[0]
        distribution_memory = torch.squeeze(distribution_memory)

        query_features = distribution_memory[:query_features.size(0), :]
        support_features = distribution_memory[query_features.size(0):, :]
        
        return class_prototipies, support_features, query_features