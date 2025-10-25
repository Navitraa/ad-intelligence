import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageFeatureExtractor:
    def __init__(self, model_name='resnet50', use_gpu=False):
        """
        Initialize the feature extractor with a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model to use (default: 'resnet50')
            use_gpu (bool): Whether to use GPU if available (default: False)
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model_name = model_name.lower()
        
        # Load pre-trained model
        self.model = self._load_pretrained_model()
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_pretrained_model(self):
        """Load pre-trained model and modify it to return features instead of classification."""
        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # Remove the last fully connected layer to get features
            model = torch.nn.Sequential(*(list(model.children())[:-1]))
        elif self.model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            # Remove the classifier to get features
            model = model.features
            model.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        return model
    
    def extract_features(self, image_path):
        """
        Extract features from an image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray: Extracted features as a 1D array
        """
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Extract features
        with torch.no_grad():
            features = self.model(image)
            
        # Flatten the features
        features = features.view(features.size(0), -1)
        return features.squeeze().cpu().numpy()
    
    def batch_extract(self, image_paths, batch_size=32):
        """
        Extract features from a batch of images.
        
        Args:
            image_paths (list): List of image paths
            batch_size (int): Batch size for processing (default: 32)
            
        Returns:
            numpy.ndarray: Array of extracted features
        """
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image = self.transform(image)
                    batch_images.append(image)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    continue
            
            if not batch_images:
                continue
                
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
                batch_features = batch_features.view(batch_features.size(0), -1)
                all_features.append(batch_features.cpu().numpy())
        
        if not all_features:
            return np.array([])
            
        return np.vstack(all_features)
