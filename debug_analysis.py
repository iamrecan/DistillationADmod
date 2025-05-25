#!/usr/bin/env python3
"""
Debug Script - SAM2 ve Google AI analiz farklarını araştırır
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from llm_image_analysis import GoogleAIImageAnalyzer

class AnalysisDebugger:
    """Analiz farklarını debug eden sınıf"""
    
    def __init__(self):
        self.analyzer = GoogleAIImageAnalyzer()
    
    def analyze_image_differences(self, original_path: str, sam2_output_path: str):
        """Orijinal ve SAM2 çıktısı arasındaki farkları analiz eder"""
        
        print("🔍 GÖRÜNTÜ FARK ANALİZİ")
        print("=" * 50)
        
        # Görüntüleri yükle
        original = cv2.imread(original_path)
        sam2_output = cv2.imread(sam2_output_path)
        
        if original is None:
            print(f"❌ Orijinal görüntü yüklenemedi: {original_path}")
            return
        
        if sam2_output is None:
            print(f"❌ SAM2 çıktısı yüklenemedi: {sam2_output_path}")
            return
        
        print(f"📁 Orijinal: {original_path}")
        print(f"📁 SAM2 Çıktı: {sam2_output_path}")
        
        # Görüntü bilgileri
        print(f"\n📊 Orijinal boyut: {original.shape}")
        print(f"📊 SAM2 çıktı boyut: {sam2_output.shape}")
        
        # Her iki görüntüyü ayrı ayrı analiz et
        print("\n" + "="*50)
        print("🔬 ORİJİNAL GÖRÜNTÜ ANALİZİ")
        print("="*50)
        
        original_analysis = self.analyzer.analyze_image(
            original_path, 
            "wood_inspection",
            custom_prompt="""Bu ahşap görüntüsünde defekt analizi yapın. Özellikle DİKKAT EDİN:

1. **Delikler (Holes)**: Ahşapta açık veya koyu renkli delik şeklindeki defektler
2. **Budaklar (Knots)**: Oval veya yuvarlak şekilli doğal defektler
3. **Çatlaklar (Cracks)**: İnce çizgi şeklindeki yarıklar
4. **Renk değişiklikleri**: Lekeli veya farklı tonlarda alanlar

Bu görüntü "hole" (delik) klasöründen geldiği için muhtemelen delik defekti içeriyor.
DİKKATLİCE inceleyin ve TÜM defektleri tespit edin.

JSON formatında yanıt verin:
{
  "defektler": {
    "delik": {"var": true/false, "açıklama": "detay", "konum": "nerede"},
    "budak": {"var": true/false, "açıklama": "detay", "konum": "nerede"},
    "çatlak": {"var": true/false, "açıklama": "detay"},
    "renk_değişikliği": {"var": true/false, "açıklama": "detay"}
  },
  "genel_değerlendirme": "görüntüdeki ana problemler neler?"
}"""
        )
        
        if original_analysis.get("success"):
            print("✅ Orijinal analiz başarılı")
            self.print_analysis_result(original_analysis.get("analysis", {}))
        else:
            print(f"❌ Orijinal analiz başarısız: {original_analysis.get('error')}")
        
        print("\n" + "="*50)
        print("🎯 SAM2 ÇIKTISI ANALİZİ")
        print("="*50)
        
        sam2_analysis = self.analyzer.analyze_image(
            sam2_output_path,
            "wood_inspection", 
            custom_prompt="""Bu SAM2 segmentasyon çıktısını analiz edin:

1. **Segmentasyon kalitesi**: Sınırlar doğru çizilmiş mi?
2. **Eksik segmentler**: Gözden kaçan alanlar var mı?
3. **Yanlış segmentler**: Hatalı işaretlenen bölgeler var mı?
4. **Defekt tespiti**: Segmentasyon defektleri doğru yakalamış mı?

Bu görüntü "hole" defekti içeren bir ahşap yüzeyinin SAM2 çıktısıdır.
Segmentasyon delik defektini yakalamış mı kontrol edin.

JSON formatında yanıt verin:
{
  "segmentasyon_kalitesi": "iyi/orta/kötü",
  "tespit_edilen_defektler": ["liste"],
  "eksik_segmentler": "var mı?",
  "segmentasyon_hatası": "analiz",
  "öneriler": ["iyileştirme önerileri"]
}"""
        )
        
        if sam2_analysis.get("success"):
            print("✅ SAM2 çıktı analizi başarılı")
            self.print_analysis_result(sam2_analysis.get("analysis", {}))
        else:
            print(f"❌ SAM2 çıktı analizi başarısız: {sam2_analysis.get('error')}")
        
        # Karşılaştırmalı analiz
        print("\n" + "="*50)
        print("⚖️ KARŞILAŞTIRMALI ANALİZ")
        print("="*50)
        
        comparison_analysis = self.analyzer.analyze_segmented_image(
            original_path,
            sam2_output_path,
            "wood_inspection"
        )
        
        if comparison_analysis.get("success"):
            print("✅ Karşılaştırmalı analiz başarılı")
            self.print_analysis_result(comparison_analysis.get("analysis", {}))
        else:
            print(f"❌ Karşılaştırmalı analiz başarısız: {comparison_analysis.get('error')}")
        
        # Debug raporu oluştur
        debug_report = {
            "original_analysis": original_analysis,
            "sam2_analysis": sam2_analysis,
            "comparison_analysis": comparison_analysis,
            "paths": {
                "original": original_path,
                "sam2_output": sam2_output_path
            }
        }
        
        debug_file = "debug_analysis_report.json"
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(debug_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Debug raporu kaydedildi: {debug_file}")
        
        return debug_report
    
    def print_analysis_result(self, analysis):
        """Analiz sonucunu yazdır"""
        if isinstance(analysis, dict):
            for key, value in analysis.items():
                if isinstance(value, dict):
                    print(f"📋 {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"   {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    print(f"📋 {key}: {', '.join(value) if value else 'Yok'}")
                else:
                    print(f"📋 {key}: {value}")
        else:
            print(f"📋 Sonuç: {analysis}")
    
    def check_sam2_prompts(self, original_path: str):
        """SAM2 prompt'larının uygunluğunu kontrol eder"""
        
        print("\n🎯 SAM2 PROMPT UYGUNLUK KONTROLÜ")
        print("=" * 50)
        
        # Görüntü boyutlarını al
        img = cv2.imread(original_path)
        if img is None:
            print("❌ Görüntü yüklenemedi")
            return
        
        h, w = img.shape[:2]
        print(f"📊 Görüntü boyutları: {w}x{h}")
        
        # Mevcut prompt koordinatlarını kontrol et (rapordan)
        center_x, center_y = 512, 512  # Rapordan alındı
        
        print(f"\n🎯 Kullanılan prompt koordinatları:")
        print(f"   Merkez nokta: ({center_x}, {center_y})")
        print(f"   Diğer noktalar: ({center_x-50}, {center_y-50}), ({center_x+50}, {center_y+50})")
        
        # Koordinat uygunluğunu kontrol et
        if center_x > w or center_y > h:
            print("⚠️ UYARI: Prompt koordinatları görüntü sınırlarının dışında!")
            print(f"   Görüntü boyutu: {w}x{h}")
            print(f"   Prompt koordinatı: {center_x}x{center_y}")
        
        # Defekt bölgesini tahmin et
        print(f"\n🔍 Defekt bölgesi analizi için öneriler:")
        
        # Daha iyi prompt önerileri
        suggested_prompts = []
        
        # Görüntüyü 9 bölgeye böl
        for i in range(3):
            for j in range(3):
                x = int(w * (j + 0.5) / 3)
                y = int(h * (i + 0.5) / 3)
                suggested_prompts.append({"x": x, "y": y, "label": True})
        
        print(f"   Önerilen çoklu nokta stratejisi:")
        for i, prompt in enumerate(suggested_prompts):
            print(f"   {i+1}. ({prompt['x']}, {prompt['y']})")
        
        return suggested_prompts

def main():
    """Ana fonksiyon"""
    
    print("🐛 SAM2 + Google AI Debug Analyzer")
    print("=" * 50)
    
    # Mevcut rapordaki dosya yollarını kullan
    original_path = "dataset/wood/test/hole/000.png"
    sam2_output_path = "output_folder/out.jpg"  # output_folder'dan al
    
    if not os.path.exists(original_path):
        print(f"❌ Orijinal dosya bulunamadı: {original_path}")
        return
    
    if not os.path.exists(sam2_output_path):
        print(f"❌ SAM2 çıktı dosyası bulunamadı: {sam2_output_path}")
        
        # Alternatif konumları kontrol et
        alternative_paths = [
            "temp_analysis/sam2_output_1748170015/out.jpg",
            "out.jpg",
            "output_folder/out.jpg"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                sam2_output_path = alt_path
                print(f"   ✅ Bulunan çıktı: {sam2_output_path}")
                break
        else:
            print("❌ Hiçbir SAM2 çıktısı bulunamadı.")
            print("   Mevcut dosyalar:")
            if os.path.exists("output_folder"):
                files = os.listdir("output_folder")
                for file in files:
                    print(f"     - {file}")
            return
    
    try:
        debugger = AnalysisDebugger()
        
        # Ana debug analizi
        debug_results = debugger.analyze_image_differences(original_path, sam2_output_path)
        
        # Prompt uygunluk kontrolü
        suggested_prompts = debugger.check_sam2_prompts(original_path)
        
        print("\n" + "="*60)
        print("📋 DEBUG SONUÇLARI ÖZETİ")
        print("="*60)
        
        print("1. Orijinal görüntüyü tek başına analiz ettik")
        print("2. SAM2 çıktısını tek başına analiz ettik") 
        print("3. İkisini karşılaştırmalı olarak analiz ettik")
        print("4. SAM2 prompt koordinatlarını kontrol ettik")
        
        print(f"\n💡 ÖNERİLER:")
        print(f"   • SAM2 prompt'larını defekt bölgesine odaklayın")
        print(f"   • Çoklu nokta stratejisi kullanın")
        print(f"   • Google AI prompt'ını daha spesifik hale getirin")
        print(f"   • Segmentasyon kalitesini manuel olarak kontrol edin")
        
    except Exception as e:
        print(f"❌ Debug analizi sırasında hata: {e}")

if __name__ == "__main__":
    main()