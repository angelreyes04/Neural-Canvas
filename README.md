# Neural-Canvas

# StyleNet Overview
Our neural style transfer implementation uses a VGG19-based feature extractor to capture content and style information from images.

Takes a content image
Uses layer 'conv4_2' of VGG19 to extract content features
These features preserve the structure and arrangement of the image

Takes 3 style images
Uses layers 'conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1' to extract style features
Computes Gram matrices to capture style and texture information
Combines features using weighted average of the styles