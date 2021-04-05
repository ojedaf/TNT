import torch
import torch.nn as nn
from collections import OrderedDict
from .utils import split_first_dim_linear
from .config_networks import ConfigureNetworks
from .set_encoder import mean_pooling
import torch.nn.functional as F
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csgraph
from transformers import RobertaTokenizer
from sentence_transformers import SentenceTransformer
import pickle as pkl

torch.autograd.set_detect_anomaly(True)

NUM_SAMPLES=1

class SimpleTSMCnaps(nn.Module):
    """
    Main model class. Implements several Simple CNAPs models (with / without feature adaptation, with /without auto-regressive
    adaptation parameters generation.
    :param device: (str) Device (gpu or cpu) on which model resides.
    :param use_two_gpus: (bool) Whether to paralleize the model (model parallelism) across two GPUs.
    :param args: (Argparser) Arparse object containing model hyper-parameters.
    """
    def __init__(self, device, conf):
        super(SimpleTSMCnaps, self).__init__()

        self.device = device

        self.feature_adaptation = conf['feature_adaptation']
        self.CLASS_NUM = conf['class_num']
        self.SAMPLE_NUM_PER_CLASS = conf['sample_num_per_class']
        self.BATCH_NUM_PER_CLASS = conf['batch_num_per_class']
        self.distance = conf['distance_metric'] # Coseine, Euclidean, Mahalanobis Distance

        if self.distance == 'coseine':
            self.cosine_distance = nn.CosineSimilarity(dim=2, eps=1e-6)

        conf_tsm = conf['TSM']
        path_checkpoint_feature_encoder = conf['checkpoints']['path_feature_encoder']
        self.text_emb = conf['text']['text_emb']
        conf_text = conf['text']

        self.type_encoder = conf['type_encoder']
        conf_class_representation_module = conf['class_representation_module']
        conf_class_representation_module['sample_num_per_class'] = conf['sample_num_per_class']
        networks = ConfigureNetworks(feature_adaptation=self.feature_adaptation,
                                     conf = conf_tsm, conf_text = conf_text, type_encoder = self.type_encoder, path_checkpoint_feature_encoder = path_checkpoint_feature_encoder, conf_class_repre = conf_class_representation_module)

        self.set_encoder = networks.get_encoder()

        self.class_representation_module = conf_class_representation_module['active']
        if self.class_representation_module == True:
            print("Class representation encoder included")
            self.att_class_repre = networks.get_class_representation_encoder()
            
        # self.temp_att_class_repre = conf_class_representation_module['active'] and conf_class_representation_module['temp_att']

        if self.type_encoder != 'video_encoder':
            if self.text_emb == 'word_level':
                self.conf_text = conf_text
                if 'text_encoder' in conf_text and conf_text['text_encoder'] == 'glove':
                    self.dict_embedding_glove = {}
                    with open(conf_text['path_glove'], 'rb') as f:
                        self.dict_embedding_glove = pkl.load(f)
                else:
                    self.max_length_text = conf['text']['temp_dim'] # 40 max num words in the train set
                    self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            else:
                if 'text_encoder' in conf_text and conf_text['text_encoder'] == 'roberta_large':
                    self.text_encoder = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
                else:
                    self.text_encoder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

        
        self.pooling_temp = conf_tsm['pooling']
        if self.pooling_temp == 'attention':
            print('Temporal pooling module was added')
            self.att_module = networks.get_attention_module()

        """
        SCM: since Simple CNAPS relies on the Mahalanobis distance, it doesn't require
        the classification adaptation function and the classifier itself. The removal of the former
        results in a 788,485 reduction in the number of parameters in the model.
        """
        #self.classifier_adaptation_network = networks.get_classifier_adaptation()
        #self.classifier = networks.get_classifier()
        
        self.feature_extractor = networks.get_feature_extractor()
        self.feature_adaptation_network = networks.get_feature_adaptation()
        self.task_representation = None
        self.graph_node_values = None
        self.class_representations = OrderedDict()  # Dictionary mapping class label (integer) to encoded representation
        
        """
        SCM: in addition to saving class representations, Simple CNAPS uses a separate
        ordered dictionary for saving the class percision matrices for use when infering on
        query examples.
        """
        self.class_precision_matrices = OrderedDict() # Dictionary mapping class label (integer) to regularized precision matrices estimated

    def getGloveEmb(self, sentences):
        list_emb = []
        for sentence in sentences:
            sentence = sentence.replace(',','')
            words = sentence.split(' ')
            emb = self.dict_embedding_glove[words[0]]
            for i in range(1, len(words)):
                emb += self.dict_embedding_glove[words[i]]
            emb /= len(words)
            list_emb.append(torch.from_numpy(emb))
        return torch.stack(list_emb).to(self.device)
    
    def forward(self, support_videos, support_labels, support_label_text, query_videos):
        """
        Forward pass through the model for one episode.
        :param support_videos: (torch.tensor) Images in the context set (batch x TC x H x W).
        :param support_labels: (torch.tensor) Labels for the context set (batch x 1 -- integer representation).
        :param query_videos: (torch.tensor) Images in the target set (batch x TC x H x W).
        :return: (torch.tensor) Categorical distribution on label set for each image in target set (batch x num_labels).
        """

        # extract train and test features

        support_videos = support_videos.to(self.device)
        query_videos = query_videos.to(self.device)

        if self.type_encoder == 'text_encoder':
            if self.text_emb == 'word_level': 
                if 'text_encoder' in self.conf_text and self.conf_text['text_encoder'] == 'glove':
                    support_text_emb = self.getGloveEmb(support_label_text)
                    self.task_representation, task_representation_by_elem = self.set_encoder(support_text_emb)
                else:
                    tokens = self.tokenizer(support_label_text, return_tensors="pt", padding=True, max_length=self.max_length_text)['input_ids']
                    support_tokens = torch.ones(tokens.size(0), self.max_length_text).long()
                    support_tokens[:,:tokens.size(1)] = tokens
                    support_tokens = support_tokens.to(self.device)
                    self.task_representation = self.set_encoder(support_tokens)
            else:
                text_emb_sentence_level = self.text_encoder.encode(support_label_text)
                support_text_emb = torch.from_numpy(text_emb_sentence_level).to(self.device)
                self.task_representation, task_representation_by_elem = self.set_encoder(support_text_emb)
                
        elif self.type_encoder == 'video_encoder':
            self.task_representation, task_representation_by_elem = self.set_encoder(support_videos)

        else:
            if self.text_emb == 'word_level': 
                tokens = self.tokenizer(support_label_text, return_tensors="pt", padding=True, max_length=self.max_length_text)['input_ids']
                support_tokens = torch.ones(tokens.size(0), self.max_length_text).long()
                support_tokens[:,:tokens.size(1)] = tokens
                support_tokens = support_tokens.to(self.device)
                self.task_representation = self.set_encoder(support_videos, support_tokens)
            else:
                text_emb_sentence_level = self.text_encoder.encode(support_label_text)
                support_text_emb = torch.from_numpy(text_emb_sentence_level).to(self.device)
                self.task_representation, task_representation_by_elem = self.set_encoder(support_videos, support_text_emb)


        # support_view_image = support_videos.view((-1, 3) + support_videos.size()[-2:])
        # self.task_representation = self.set_encoder(support_view_image)

        support_features, query_features = self._get_features(support_videos, query_videos)

        if self.class_representation_module == False:
            if self.pooling_temp == 'attention':
                support_features = self.att_module(support_features)
                query_features = self.att_module(query_features)
            else:
                support_features = torch.mean(support_features, dim = 1)
                query_features = torch.mean(query_features, dim = 1)

        if self.distance == 'mahalanobis':
            """
            SCM: we build both class representations and the regularized covariance estimates.
            """
            # get the class means and covariance estimates in tensor form

            if self.class_representation_module == True:
                class_prototipies, support_features, query_features = self.att_class_repre(query_features, support_features, task_representation_by_elem)
                self._build_att_class_reps_and_covariance_estimates(support_features, support_labels, class_prototipies)
            else:    
                self._build_class_reps_and_covariance_estimates(support_features, support_labels)
            
            class_means = torch.stack(list(self.class_representations.values())).squeeze(1)
            class_precision_matrices = torch.stack(list(self.class_precision_matrices.values()))

            # grabbing the number of classes and query examples for easier use later in the function
            number_of_classes = class_means.size(0)
            number_of_targets = query_features.size(0)

            """
            SCM: calculating the Mahalanobis distance between query examples and the class means
            including the class precision estimates in the calculations, reshaping the distances
            and multiplying by -1 to produce the sample logits
            """
            repeated_query = query_features.repeat(1, number_of_classes).view(-1, class_means.size(1))
            repeated_class_means = class_means.repeat(number_of_targets, 1)
            repeated_difference = (repeated_class_means - repeated_query)
            repeated_difference = repeated_difference.view(number_of_targets, number_of_classes, repeated_difference.size(1)).permute(1, 0, 2)
            first_half = torch.matmul(repeated_difference, class_precision_matrices)
            sample_logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1,0) * -1

            # clear all dictionaries
            self.class_representations.clear()
            self.class_precision_matrices.clear()

            return split_first_dim_linear(sample_logits, [NUM_SAMPLES, query_videos.shape[0]])
        else:

            support_features = support_features.view(self.CLASS_NUM,self.SAMPLE_NUM_PER_CLASS,support_features.size(1))
            support_features = torch.sum(support_features,1).squeeze(1)

            size_support = support_features.size()
            support_features = support_features.view(size_support[0], -1)
            size_support = support_features.size()
            support_features = torch.unsqueeze(support_features, dim = 0).expand(self.BATCH_NUM_PER_CLASS*self.CLASS_NUM, size_support[0], size_support[1])

            size_query = query_features.size()
            query_features = query_features.view(size_query[0], -1)
            size_query = query_features.size()
            query_features = torch.unsqueeze(query_features, dim = 1).expand(size_query[0], self.CLASS_NUM, size_query[1])

            if self.distance == 'coseine':
                return self.cosine_distance(query_features, support_features)
            else:
                return -torch.norm(query_features - support_features, dim=2, p=2)

    def _get_features(self, support_videos, query_videos):
        """
        Helper function to extract task-dependent feature representation for each image in both context and target sets.
        :param context_images: (torch.tensor) Images in the context set (batch x T x C x H x W).
        :param target_images: (torch.tensor) Images in the target set (batch x T x C x H x W).
        :return: (tuple::torch.tensor) Feature representation for each set of images.
        """

        if self.feature_adaptation == 'film+ar':
            # Get adaptation params by passing context set through the adaptation networks
            self.feature_extractor_params = self.feature_adaptation_network(support_videos.view((-1, 3) + support_videos.size()[-2:]), self.task_representation)
        else:
            # Get adaptation params by passing context set through the adaptation networks
            self.feature_extractor_params = self.feature_adaptation_network(self.task_representation)
        # Given adaptation parameters for task, conditional forward pass through the adapted feature extractor
        support_features = self.feature_extractor(support_videos, self.feature_extractor_params)
        query_features = self.feature_extractor(query_videos, self.feature_extractor_params)

        return support_features, query_features

    def _build_class_reps_and_covariance_estimates(self, support_features, support_labels):
        """
        Construct and return class level representations and class covariance estimattes for each class in task.
        :param support_features: (torch.tensor) Adapted feature representation for each image in the context set.
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation and class covariance estimates dictionary.
        """

        """
        SCM: calculating a task level covariance estimate using the provided function.
        """
        task_covariance_estimate = self.estimate_cov(support_features)

        support_features = support_features.view(self.CLASS_NUM,self.SAMPLE_NUM_PER_CLASS,support_features.size(1))

        b_I = torch.eye(support_features.size(2), support_features.size(2)).to(self.device)

        for c in range(self.CLASS_NUM):
            # filter out feature vectors which have class c
            class_features = support_features[c, :, :]
            # mean pooling examples to form class means
            class_rep = mean_pooling(class_features)
            # updating the class representations dictionary with the mean pooled representation
            self.class_representations[c] = class_rep
            """
            Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
            Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
            inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
            dictionary for use later in infering of the query data points.
            """
            lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
            self.class_precision_matrices[c] = torch.inverse((lambda_k_tau * self.estimate_cov(class_features)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
                    + b_I)
    
    def estimate_cov(self, examples, rowvar=False, inplace=False):
        """
        SCM: unction based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

        Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        """

        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()

    def _build_att_class_reps_and_covariance_estimates(self, support_features, support_labels, class_prototipies):
        """
        Construct and return class level representations and class covariance estimattes for each class in task.
        :param support_features: (torch.tensor) Adapted feature representation for each image in the context set.
        :param context_labels: (torch.tensor) Label for each image in the context set.
        :return: (void) Updates the internal class representation and class covariance estimates dictionary.
        """

        """
        SCM: calculating a task level covariance estimate using the provided function.
        """

        general_prototipe = torch.mean(class_prototipies, dim=0, keepdim=True)
        task_covariance_estimate = self.estimate_cov_att(support_features, general_prototipe)

        support_features = support_features.view(self.CLASS_NUM,self.SAMPLE_NUM_PER_CLASS,support_features.size(1))

        b_I = torch.eye(support_features.size(2), support_features.size(2)).to(self.device)

        for c in range(self.CLASS_NUM):
            # filter out feature vectors which have class c
            class_features = support_features[c, :, :]
            # mean pooling examples to form class means
            class_rep = torch.unsqueeze(class_prototipies[c, :], dim=0)
            # updating the class representations dictionary with the mean pooled representation
            self.class_representations[c] = class_rep
            """
            Calculating the mixing ratio lambda_k_tau for regularizing the class level estimate with the task level estimate."
            Then using this ratio, to mix the two estimate; further regularizing with the identity matrix to assure invertability, and then
            inverting the resulting matrix, to obtain the regularized precision matrix. This tensor is then saved in the corresponding
            dictionary for use later in infering of the query data points.
            """
            lambda_k_tau = (class_features.size(0) / (class_features.size(0) + 1))
            self.class_precision_matrices[c] = torch.inverse((lambda_k_tau * self.estimate_cov_att(class_features, class_rep)) + ((1 - lambda_k_tau) * task_covariance_estimate) \
                    + b_I)
    
    def estimate_cov_att(self, examples, prototipies, rowvar=False, inplace=False):
        """
        SCM: unction based on the suggested implementation of Modar Tensai
        and his answer as noted in: https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

        Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        """

        if examples.dim() > 2:
            raise ValueError('m has more than 2 dimensions')
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
            prototipies = prototipies.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= prototipies
        else:
            examples = examples - prototipies
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()

    @staticmethod
    def _extract_class_indices(labels, which_class):
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    def distribute_model(self):
        # self.feature_extractor.cuda(1)
        # self.feature_adaptation_network.cuda(1)

        if torch.cuda.device_count() > 1:
            self.feature_extractor = nn.DataParallel(self.feature_extractor)
            self.feature_adaptation_network = nn.DataParallel(self.feature_adaptation_network)
            self.set_encoder = nn.DataParallel(self.set_encoder)
        
        print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)

        self.feature_extractor.to(self.device)
        self.feature_adaptation_network.to(self.device)
        self.set_encoder.to(self.device)

        if self.class_representation_module == True:
            self.att_class_repre.to(self.device)
        
        if self.pooling_temp == 'attention':
            self.att_module.to(self.device)
            