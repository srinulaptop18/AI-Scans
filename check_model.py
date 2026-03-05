"""
Model Architecture Checker
==========================
This script helps you identify which architecture your model uses.
Run this before running the main app to debug issues.
"""

import torch
import os

def check_model_file(filename):
    """Check what architecture a model file contains"""
    
    if not os.path.exists(filename):
        print(f"❌ File '{filename}' not found!")
        return
    
    print(f"\n{'='*60}")
    print(f"Checking: {filename}")
    print('='*60)
    
    try:
        # Load the model
        checkpoint = torch.load(filename, map_location='cpu')
        
        print(f"\n✅ File loaded successfully!")
        print(f"📦 Type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            # It's a state_dict
            keys = list(checkpoint.keys())
            print(f"\n🔑 Number of keys: {len(keys)}")
            print(f"\n📋 First 10 keys:")
            for key in keys[:10]:
                shape = checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else 'N/A'
                print(f"   - {key}: {shape}")
            
            # Detect architecture
            print(f"\n🔍 Architecture Detection:")
            
            has_efficientnet = any('efficientnet' in key.lower() for key in keys)
            has_mobilenet = any('mobilenet' in key.lower() for key in keys)
            has_backbone = any('backbone' in key for key in keys)
            has_transformer = any('transformer' in key for key in keys)
            has_classifier = any('classifier' in key for key in keys)
            has_head = any('head' in key for key in keys)
            
            print(f"   - Contains 'efficientnet': {has_efficientnet}")
            print(f"   - Contains 'mobilenet': {has_mobilenet}")
            print(f"   - Contains 'backbone': {has_backbone}")
            print(f"   - Contains 'transformer': {has_transformer}")
            print(f"   - Contains 'classifier': {has_classifier}")
            print(f"   - Contains 'head': {has_head}")
            
            print(f"\n🎯 Predicted Architecture:")
            if has_efficientnet and has_mobilenet:
                print("   ✅ EfficientNet-MobileNet")
            elif has_efficientnet or has_mobilenet:
                print("   🤔 Possibly EfficientNet-MobileNet (only one component detected)")
            elif has_backbone and has_transformer:
                print("   ✅ ResNet-ViT")
            elif has_classifier:
                print("   🤔 Contains classifier - likely EfficientNet-MobileNet")
            elif has_head:
                print("   🤔 Contains head - likely ResNet-ViT")
            else:
                print("   ❓ Unknown architecture - manual inspection needed")
            
            # Size info
            total_params = sum(p.numel() for p in checkpoint.values() if hasattr(p, 'numel'))
            print(f"\n📊 Total parameters: {total_params:,}")
            print(f"📏 File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
            
        else:
            print(f"\n⚠️ This is a full model (not state_dict)")
            print(f"   You may need to extract state_dict first")
    
    except Exception as e:
        print(f"\n❌ Error loading model: {str(e)}")
    
    print(f"\n{'='*60}\n")

def main():
    print("\n" + "="*60)
    print("🔍 MODEL ARCHITECTURE CHECKER")
    print("="*60)
    
    # Check both model files
    files_to_check = ['new.pth', 'old.pth']
    
    for filename in files_to_check:
        check_model_file(filename)
    
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS:")
    print("="*60)
    print("""
    If your model is EfficientNet-MobileNet:
    - Make sure 'new.pth' contains efficientnet and mobilenet keys
    - The app will detect it automatically
    
    If your model is ResNet-ViT:
    - Make sure it has 'backbone' and 'transformer' keys
    - Rename to 'old.pth' if you want to keep both
    
    If detection fails:
    - Check the key names above
    - The model architecture must match the training code
    - You may need to adjust the model class definition
    """)
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
