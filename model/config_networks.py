from .resnet_our import film_resnet18, resnet18, resnet34, film_resnet34, resnet50, film_resnet50
from .adaptation_networks import NullFeatureAdaptationNetwork, FilmAdaptationNetwork, \
    LinearClassifierAdaptationNetwork, FilmLayerNetwork, FilmArAdaptationNetwork
# from .set_encoder_text import SetEncoder
from .utils import linear_classifier
from .class_representation_encoder import ClassRepresentationEncoder, AttentionBasedRepresentationEncoder, TwoLevelAttentionBasedRepresentationEncoder, LabelPropagation
from .att_module import TimeAttention

from .temporalShiftModule.ops.models import TSN
import torch.nn as nn
import os
import torch
""" Creates the set encoder, feature extractor, feature adaptation, classifier, and classifier adaptation networks.
"""


class ConfigureNetworks:
    def __init__(self, feature_adaptation, conf, conf_text, type_encoder = 'text_encoder', path_checkpoint_feature_encoder = None, conf_class_repre = None):

        self.classifier = linear_classifier

        if conf['arch'] == 'resnet18':
            num_maps_per_layer = [64, 128, 256, 512]
            num_blocks_per_layer = [2, 2, 2, 2]
            num_initial_conv_maps = 64
            base_model_func = resnet18
            base_model_film_func = film_resnet18
            type_block = 'basicblock'
            output_dim_model = 512
        elif conf['arch'] == 'resnet34':
            num_maps_per_layer = [64, 128, 256, 512]
            num_blocks_per_layer = [3, 4, 6, 3]
            num_initial_conv_maps = 64
            base_model_func = resnet34
            base_model_film_func = film_resnet34
            type_block = 'basicblock'
            output_dim_model = 512
        elif conf['arch'] == 'resnet50':
            num_maps_per_layer = [[64,64,256], [128,128,512], [256,256,1024], [512,512,2048]]
            num_blocks_per_layer = [3, 4, 6, 3]
            num_initial_conv_maps = 64
            base_model_func = resnet50
            base_model_film_func = film_resnet50
            type_block = 'bottleneck'
            output_dim_model = 2048

        if type_encoder == 'text_encoder':
            from .set_encoder_text import SetEncoder

            text_emb = conf_text['text_emb']
            num_layers = conf_text['num_layers']
            ratio = conf_text['ratio']
            temp_dim = conf_text['temp_dim']
            embedding_size = conf_text['embedding_size']
            text_encoder = conf_text['text_encoder'] if 'text_encoder' in conf_text else 'roberta_base'

            self.encoder = SetEncoder(text_emb, num_layers, ratio, temp_dim, embedding_size, output_dim_model, text_encoder)

            print('The text encoder was added')
        elif type_encoder == 'video_encoder':
            from .set_encoder import SetEncoder

            self.encoder = SetEncoder(conf_class_repre['active'], output_dim_model)
            
            print('The video encoder was added')
        
        else:
            from .set_encoder_text_video import SetVideoTextEncoder

            text_emb = conf_text['text_emb']
            num_layers = conf_text['num_layers']
            ratio = conf_text['ratio']
            temp_dim = conf_text['temp_dim']
            embedding_size = conf_text['embedding_size']

            self.encoder = SetVideoTextEncoder(text_emb, num_layers, ratio, temp_dim, embedding_size)
            print('The video-text encoder was added')


        z_g_dim = self.encoder.pre_pooling_fn.output_size
        
        conf['path_checkpoint_feature_encoder'] = path_checkpoint_feature_encoder

        if feature_adaptation == "no_adaptation":
#             base_model = base_model_func(
#                 pretrained=True,
#                 pretrained_model_path=pretrained_resnet_path
#             )
            base_model = base_model_func(
                pretrained=True, progress=True
            )
            self.feature_extractor = self.get_tsm(conf, base_model)
            self.feature_adaptation_network = NullFeatureAdaptationNetwork()

        elif feature_adaptation == "film":
#             base_model = base_model_film_func(
#                 pretrained=True,
#                 pretrained_model_path=pretrained_resnet_path
#             )
            base_model = base_model_film_func(
                pretrained=True, progress=True
            )
            self.feature_extractor = self.get_tsm(conf, base_model)
            self.feature_adaptation_network = FilmAdaptationNetwork(
                layer=FilmLayerNetwork,
                num_maps_per_layer=num_maps_per_layer,
                num_blocks_per_layer=num_blocks_per_layer,
                z_g_dim=z_g_dim,
                type_block=type_block
            )

        elif feature_adaptation == 'film+ar':
            # Ojo aqui puede ocurrir un error al pasar el feature extractor al FilmArAdaptationNetwork mas exactamente en la linea 329 del adaptation_networks.py
#             base_model = base_model_film_func(
#                 pretrained=True,
#                 pretrained_model_path=pretrained_resnet_path
#             )
            base_model = base_model_film_func(
                pretrained=True, progress=True
            )
            self.feature_extractor = self.get_tsm(conf, base_model)
            self.feature_adaptation_network = FilmArAdaptationNetwork(
                feature_extractor=self.feature_extractor,
                num_maps_per_layer=num_maps_per_layer,
                num_blocks_per_layer=num_blocks_per_layer,
                num_initial_conv_maps = num_initial_conv_maps,
                z_g_dim=z_g_dim
            )
        
        # if path_checkpoint_feature_encoder != None and path_checkpoint_feature_encoder != '' and os.path.exists(path_checkpoint_feature_encoder):
        #     # model_dict = self.feature_extractor.state_dict()
        #     state_dict = torch.load(path_checkpoint_feature_encoder)
        #     # list_key_model = list(model_dict.keys())
        #     # state_dict = {list_key_model[i]: v for i, (k, v) in  enumerate(state_dict.items())}
        #     state_dict = state_dict['state_dict']
        #     state_dict = {k.replace('module.',''): v for k, v in  state_dict.items()}
        #     self.feature_extractor.load_state_dict(state_dict)
        #     print("load feature encoder success")

        # Freeze the parameters of the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        output_size = 512
        self.classifier_adaptation_network = LinearClassifierAdaptationNetwork(self.feature_extractor.base_model.output_size)

        if conf_class_repre != None and conf_class_repre['active'] == True and conf_class_repre['temp_att'] == False:
            if 'type' in conf_class_repre and conf_class_repre['type'] == 'label_propagation':
                self.class_repre_encoder = LabelPropagation(conf['num_segments'], conf_class_repre['sample_num_per_class'])
            else:
                self.class_repre_encoder = AttentionBasedRepresentationEncoder(self.feature_extractor.base_model.output_size, z_g_dim, conf_class_repre['num_heads'], conf['pooling'], conf['num_segments'], conf_class_repre['sample_num_per_class'])
        elif conf_class_repre != None and conf_class_repre['active'] == True and conf_class_repre['temp_att'] == True:
            self.class_repre_encoder = TwoLevelAttentionBasedRepresentationEncoder(self.feature_extractor.base_model.output_size, z_g_dim, conf_class_repre['num_heads'], conf_class_repre['sample_num_per_class'])
        else:
            self.class_repre_encoder = None
        
        if conf['pooling'] == 'attention':
            self.attention_module = TimeAttention(conf['num_segments'])
    
    def get_tsm(self, conf, model_instance):
        TSM = TSN(conf['num_class'], conf['num_segments'], conf['modality'],
                base_model=conf['arch'],
                model_instance=model_instance,
                consensus_type=conf['consensus_type'],
                dropout=conf['dropout'],
                img_feature_dim=conf['img_feature_dim'],
                partial_bn=not conf['no_partialbn'],
                pretrain=conf['pretrain'],
                is_shift=conf['shift'], shift_div=conf['shift_div'], shift_place=conf['shift_place'],
                fc_lr5=conf['fc_lr5'],
                temporal_pool=conf['temporal_pool'],
                non_local=conf['non_local'], get_emb = True)

        path_checkpoint_feature_encoder = conf['path_checkpoint_feature_encoder']
        if path_checkpoint_feature_encoder != None and path_checkpoint_feature_encoder != '' and os.path.exists(path_checkpoint_feature_encoder):
            # model_dict = self.feature_extractor.state_dict()
            try:
                state_dict = torch.load(path_checkpoint_feature_encoder)
            except:
                import tarfile
                with tarfile.open(path_checkpoint_feature_encoder, 'r') as t:
                    print(t.getnames())
            # list_key_model = list(model_dict.keys())
            # state_dict = {list_key_model[i]: v for i, (k, v) in  enumerate(state_dict.items())}
            state_dict = state_dict['state_dict']
            state_dict = {k.replace('module.',''): v for k, v in  state_dict.items() if 'base_model.fc' not in k}
            TSM.load_state_dict(state_dict)
            print("load feature encoder success")

        TSM.base_model.fc = nn.Identity()
        return TSM

    def get_encoder(self):
        return self.encoder

    def get_classifier(self):
        return self.classifier

    def get_classifier_adaptation(self):
        return self.classifier_adaptation_network

    def get_feature_adaptation(self):
        return self.feature_adaptation_network

    def get_feature_extractor(self):
        return self.feature_extractor

    def get_class_representation_encoder(self):
        return self.class_repre_encoder
    
    def get_attention_module(self):
        return self.attention_module
