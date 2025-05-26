#!/usr/bin/env python3
"""
🧪 Basit test scripti - Import problemlerini debug et
"""

def test_step_by_step():
    """Adım adım import test et"""
    
    print("🔧 Testing basic Python modules...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False
        
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except Exception as e:
        print(f"❌ NumPy error: {e}")
        return False
        
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except Exception as e:
        print(f"❌ OpenCV error: {e}")
        return False
        
    try:
        from PIL import Image
        print("✅ PIL imported")
    except Exception as e:
        print(f"❌ PIL error: {e}")
        return False
        
    print("\n🔧 Testing model imports...")
    
    try:
        from models.SingleNet.trainer_sn import SnTrainer
        print("✅ SnTrainer imported")
    except Exception as e:
        print(f"❌ SnTrainer error: {e}")
        return False
        
    try:
        from sam2_google_ai_pipeline import SAM2GoogleAIPipeline
        print("✅ SAM2GoogleAIPipeline imported")
    except Exception as e:
        print(f"❌ SAM2GoogleAIPipeline error: {e}")
        return False
        
    print("\n🔧 Testing IntegratedAnomalySystem...")
    
    try:
        from integrated_anomaly_system import IntegratedAnomalySystem
        print("✅ IntegratedAnomalySystem imported")
        
        system = IntegratedAnomalySystem()
        print("✅ System initialized")
        return True
        
    except Exception as e:
        print(f"❌ IntegratedAnomalySystem error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 DEBUG TEST STARTING...")
    print("="*50)
    
    success = test_step_by_step()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")