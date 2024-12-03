import torch 
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()

        # freeze parameters since it is pretrained
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        self.model = self._create_feature_extractor()

    def _create_feature_extractor(self):
        model = nn.Sequential()

        layer_mapping = {
            'conv1_1': self.vgg[0],
            'conv2_1': self.vgg[5],
            'conv3_1': self.vgg[10],
            'conv4_1': self.vgg[19],
            'conv4_2': self.vgg[21],
            'conv5_1': self.vgg[28]
        }

        for name, layer in layer_mapping.items():
            model.add_module(name, layer)
            model.add_module(f'{name}_relu', nn.ReLU(inplace=True))
        
        return model
    
    def extract_features(self, image):
        features = {}
        x = image

        for name, layer in self.model.named_children():
            x = layer(x)

            if name in self.content_layers or name in self.style_layers:
                features[name] = x
        return features
    
    def get_content_features(self, image):
        features = self.extract_features(image)
        return {layer: features[layer] for layer in self.content_layers}
    
    def get_style_features(self, style_images, weights):
        combined_gram_matrices = {}
        
        for image, weight in zip(style_images, weights):
            features = self.extract_features(image)
            
            for layer_name in self.style_layers:
                feat = features[layer_name]
                b, c, h, w = feat.size()
                
                feat_reshaped = feat.view(b, c, -1)
                gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
                gram = gram.div(h * w * c)  # Normalize
                
                if layer_name not in combined_gram_matrices:
                    combined_gram_matrices[layer_name] = weight * gram
                else:
                    combined_gram_matrices[layer_name] += weight * gram
                        
        return combined_gram_matrices