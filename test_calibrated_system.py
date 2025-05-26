#!/usr/bin/env python3
"""
🧪 Güncellenmiş Integrated Anomaly System Test Scripti
"""

from integrated_anomaly_system import IntegratedAnomalySystem
import os

def test_calibrated_threshold():
    """Kalibre edilmiş threshold sistemini test et"""
    
    # Sistemi başlat
    print("🔧 Sistem başlatılıyor...")
    system = IntegratedAnomalySystem()
    
    # Test parametreleri
    image_path = 'dataset/wood/test/hole/001.png'
    model_type = 'sn'
    dataset = 'wood'
    
    print(f"\n🎯 TEST BAŞLIYOR")
    print(f"📸 Görüntü: {image_path}")
    print(f"🤖 Model: {model_type}")
    print(f"📊 Dataset: {dataset}")
    print("="*60)
    
    # Anomali tespiti yap (yeni kalibre edilmiş threshold ile)
    result = system.detect_anomalies(image_path, model_type, dataset)
    
    if result['success']:
        print(f"\n🎉 TEST BAŞARILI!")
        print(f"🔍 Anomali tespit edildi: {result['has_anomaly']}")
        print(f"📏 Kullanılan threshold: {result['threshold']:.6f}")
        print(f"🎯 Threshold yöntemi: {result['threshold_method']}")
        print(f"📊 Anomali oranı: {result['anomaly_ratio']:.4%}")
        print(f"🚨 Seviye: {result['severity_emoji']} {result['severity']}")
        
        if 'calibration_result' in result and result['calibration_result']['success']:
            cal = result['calibration_result']
            print(f"\n📊 KALİBRASYON BİLGİLERİ:")
            print(f"   ✅ Kalibrasyon başarılı")
            print(f"   📏 Kalibre edilmiş threshold: {cal['calibrated_threshold']:.6f}")
            print(f"   📉 False positive rate: {cal['false_positive_rate']:.1%}")
            print(f"   📸 Test edilen normal görüntü sayısı: {cal['normal_image_count']}")
    else:
        print(f"❌ HATA: {result['error']}")

if __name__ == "__main__":
    test_calibrated_threshold()