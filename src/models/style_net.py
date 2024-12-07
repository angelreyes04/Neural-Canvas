import torch 
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import config

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()

        # freeze parameters since it is pretrained
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.content_layers = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        self.layer_mapping = {
            'conv1_1': 0,    
            'conv2_1': 5,    
            'conv3_1': 10,   
            'conv4_1': 19, 
            'conv4_2': 21, 
            'conv5_1': 28 
        }

        self.model = self._create_feature_extractor()

    def _create_feature_extractor(self):
        model = nn.Sequential()
        i = 0
        current_block = 1
        current_conv = 1
        
        for layer in self.vgg.children():
            if isinstance(layer, nn.Conv2d):
                name = f'conv{current_block}_{current_conv}'
                model.add_module(name, layer)
                
                if name in self.content_layers or name in self.style_layers:
                    self.layer_mapping[name] = i
                    
                current_conv += 1
                i += 1
                
            elif isinstance(layer, nn.ReLU):
                model.add_module(f'relu{i}', layer)
                i += 1
                
            elif isinstance(layer, nn.MaxPool2d):
                model.add_module(f'pool{i}', layer)
                current_block += 1
                current_conv = 1
                i += 1
        
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
    
    def get_style_features_and_matrix(self, style_images, weights):
        combined_gram_matrices = {}
        combined_features = {}

        for image, weight in zip(style_images, weights):
            features = self.extract_features(image)
            
            for layer_name in self.style_layers:
                feat = features[layer_name]
                b, c, h, w = feat.size()
                
                if layer_name not in combined_features:
                    combined_features[layer_name] = weight * feat
                else:
                    combined_features[layer_name] += weight * feat


                feat_reshaped = feat.view(b, c, -1)
                gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
                gram = gram.div(h * w * c)  # Normalize
                
                if layer_name not in combined_gram_matrices:
                    combined_gram_matrices[layer_name] = weight * gram
                else:
                    combined_gram_matrices[layer_name] += weight * gram
                        
        return combined_features, combined_gram_matrices
    
    def get_target_matrix(self, target_image):
        features = self.extract_features(target_image)
        target_gram_matrices = {}

        for layer_name in self.style_layers:
            feat = features[layer_name]
            b,c,h,w = feat.size
            feat_reshaped = feat.view(b,c,-1)
            gram = torch.bmm(feat_reshaped, feat_reshaped.transpose(1, 2))
            gram = gram.div(h * w * c)  # Normalize
            target_gram_matrices[layer_name] = gram

        return target_gram_matrices

    #loss
    def calc_style_loss(self, s_gram, t_gram):
        style_loss = 0
        for layer_name in self.style_layers:
            style_gram = s_gram[layer_name]
            target_gram = t_gram[layer_name]
            layer_loss = torch.sum((style_gram - target_gram)**2)
            style_loss += layer_loss
        
        return style_loss
    
    def calc_content_loss(self,c_features, s_features):
        content_loss = 0
        for layer in self.content_layers:
            c_layer_features = c_features[layer]
            s_layer_features = s_features[layer]
            content_loss += 0.5 * torch.sum((c_layer_features - s_layer_features)**2)
        
        return content_loss
    
    def calc_loss(self, c_features, s_features, s_gram, c_gram, alpha, beta):
        content_loss = self.calc_content_loss(c_features, s_features)
        style_loss = self.calc_style_loss(s_gram, c_gram)
        total_loss = alpha * content_loss + beta * style_loss
        
        return total_loss
    
def optimize_image(content_image, style_images, weights):
    extractor = FeatureExtractor()
    
    generated_image = content_image.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([generated_image], lr=config.LEARNING_RATE)

    content_features = extractor.get_content_features(content_image)
    _, style_gram = extractor.get_style_features_and_matrix(style_images, weights)

    for step in range(config.NUM_STEPS):
        optimizer.zero_grad()

        generated_content = extractor.get_content_features(generated_image)
        _, generated_gram = extractor.get_style_features_and_matrix([generated_image], [1.0])

        total_loss = extractor.calc_loss(
            generated_content, 
            content_features,
            generated_gram, 
            style_gram,
            config.CONTENT_WEIGHT,
            config.STYLE_WEIGHT
        )

        total_loss.backward()
        optimizer.step()

    return generated_image