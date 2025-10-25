import os
import argparse
import numpy as np
from ad_intel.feature_extractor import ImageFeatureExtractor

def main():
    parser = argparse.ArgumentParser(description='Extract image features using a pre-trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image or directory')
    parser.add_argument('--model', type=str, default='resnet50', help='Pre-trained model to use (resnet50 or vgg16)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--output', type=str, help='Path to save the features (numpy .npy file)')
    
    args = parser.parse_args()
    
    # Initialize the feature extractor
    extractor = ImageFeatureExtractor(model_name=args.model, use_gpu=args.gpu)
    print(f"Using {args.model} for feature extraction")
    
    # Check if input is a single file or directory
    if os.path.isfile(args.image_path):
        # Single image
        print(f"Extracting features from: {args.image_path}")
        features = extractor.extract_features(args.image_path)
        print(f"Extracted features shape: {features.shape}")
        
        if args.output:
            np.save(args.output, features)
            print(f"Features saved to {args.output}")
            
    elif os.path.isdir(args.image_path):
        # Directory of images
        print(f"Processing directory: {args.image_path}")
        image_paths = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_paths:
            print("No image files found in the directory")
            return
            
        print(f"Found {len(image_paths)} images")
        features = extractor.batch_extract(image_paths)
        
        if args.output:
            np.save(args.output, features)
            print(f"Features for {len(image_paths)} images saved to {args.output}")
    else:
        print(f"Error: {args.image_path} is not a valid file or directory")
        return

if __name__ == "__main__":
    main()
