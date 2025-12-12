import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
import json
from pathlib import Path


class OutfitClassifier:
    def __init__(self, checkpoint_path, device=None):
        """Initialize classifier with trained model"""
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_names = checkpoint.get('class_names', [
            'casual_everyday', 'gym_athletic', 'interview_professional', 'party_social'
        ])
        num_classes = len(self.class_names)
        
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        print(f"Model loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")
    
    def predict(self, image_path, top_k=3):
        """
        Predict outfit category for a single image
        
        Args:
            image_path: Path to image file
            top_k: Return top-k predictions
            
        Returns:
            Dictionary with predictions and probabilities
        """
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        top_probs, top_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': self.class_names[idx.item()],
                'probability': prob.item(),
                'confidence': f"{prob.item() * 100:.2f}%"
            })
        
        return {
            'image_path': str(image_path),
            'top_prediction': predictions[0],
            'all_predictions': predictions
        }
    
    def predict_batch(self, image_paths, top_k=3):
        """Predict for multiple images"""
        results = []
        for img_path in image_paths:
            result = self.predict(img_path, top_k)
            if result:
                results.append(result)
        return results


def format_prediction_output(result):
    """Format prediction result for display"""
    if not result:
        return "No result"
    
    output = []
    output.append(f"\n{'='*60}")
    output.append(f"Image: {Path(result['image_path']).name}")
    output.append(f"{'='*60}")
    
    output.append(f"Predicted Category: {result['top_prediction']['class'].upper()}")
    output.append(f"   Confidence: {result['top_prediction']['confidence']}")
    
    if len(result['all_predictions']) > 1:
        output.append(f"All Predictions:")
        for i, pred in enumerate(result['all_predictions'], 1):
            output.append(f"   {i}. {pred['class']:<25} {pred['confidence']:>8}")
    
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description='Predict outfit category for images')
    parser.add_argument('images', nargs='+', help='Path(s) to image file(s)')
    parser.add_argument('--checkpoint', default='checkpoints/final_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Show top-k predictions')
    parser.add_argument('--output', default=None,
                        help='Save results to JSON file')
    
    args = parser.parse_args()
    
    print("Loading outfit classifier...")
    classifier = OutfitClassifier(args.checkpoint)
    
    print(f"Processing {len(args.images)} image(s)...")
    results = classifier.predict_batch(args.images, top_k=args.top_k)
    
    for result in results:
        print(format_prediction_output(result))
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    
    print(f"Processed {len(results)} image(s)")


if __name__ == "__main__":
    main()