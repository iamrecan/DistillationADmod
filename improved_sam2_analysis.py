#!/usr/bin/env python3
"""
Improved SAM2 Pipeline - Defekt odaklı segmentasyon
"""

import cv2
import numpy as np
import json
from sam2_google_ai_pipeline import SAM2GoogleAIPipeline

def create_defect_focused_prompts(image_path: str) -> dict:
    """Defekt odaklı SAM2 prompt'ları oluşturur"""
    
    # Görüntüyü yükle
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Görüntü yüklenemedi")
    
    h, w = img.shape[:2]
    
    # Wood/hole dataset için özel prompt stratejisi
    prompts = {
        "prompts": [
            # Sol üst delik için
            {
                "type": "points",
                "object_id": 1,
                "frame_index": 0,
                "points": [
                    {"x": int(w * 0.3), "y": int(h * 0.3), "label": True},  # Sol üst bölge
                    {"x": int(w * 0.25), "y": int(h * 0.25), "label": True}
                ]
            },
            # Sağ alt delik için
            {
                "type": "points", 
                "object_id": 2,
                "frame_index": 0,
                "points": [
                    {"x": int(w * 0.7), "y": int(h * 0.7), "label": True},  # Sağ alt bölge
                    {"x": int(w * 0.75), "y": int(h * 0.75), "label": True}
                ]
            },
            # Merkez bölge için (budak veya diğer defektler)
            {
                "type": "points",
                "object_id": 3, 
                "frame_index": 0,
                "points": [
                    {"x": int(w * 0.5), "y": int(h * 0.8), "label": True},  # Alt merkez (budak)
                    {"x": int(w * 0.5), "y": int(h * 0.5), "label": False}  # Negative sample
                ]
            }
        ]
    }
    
    return prompts

def create_improved_ai_prompt() -> str:
    """İyileştirilmiş Google AI prompt'u"""
    
    return """Bu görüntüde SAM2 segmentasyon sonuçlarını ÇOKCOK DİKKATLİ analiz edin:

⚠️ ÖNEMLİ: Bu görüntü "hole" (delik) defekti içeren bir ahşap yüzeyinin SAM2 segmentasyon çıktısıdır.

1. **YANLIŞ POZİTİF KONTROLÜ**:
   - Segmentasyon YANLIŞ delik işaretlemiş olabilir
   - Normal ahşap dokusu delik olarak işaretlenmiş olabilir mi?
   - Budaklar delik olarak yanlış sınıflandırılmış olabilir mi?

2. **GERÇEK DEFEKT TESPİTİ**:
   - Gerçek delikler: Açık, koyu renkli, düzensiz şekilli
   - Budaklar: Oval, gömülü, doğal doku
   - Sahte segmentler: Normal ahşap alanları

3. **KALİTE DEĞERLENDİRMESİ**:
   - Segmentasyon kalitesi nasıl?
   - Hangi defektler DOĞRU tespit edilmiş?
   - Hangi alanlar YANLIŞ işaretlenmiş?

4. **KARŞILAŞTIRMA**:
   - Orijinal görüntüde kaç defekt var?
   - SAM2 çıktısında kaç segment var?
   - Sayı uyumsuzluğu var mı?

ÇOOOOOK DİKKATLİ OLUN: Yanlış pozitifleri tespit edin!

JSON formatında yanıt verin:
{
  "segmentasyon_dogrulugu": {
    "dogru_tespitler": ["gerçek defektler"],
    "yanlis_pozitifler": ["yanlış işaretlenen alanlar"],
    "eksik_tespitler": ["gözden kaçan defektler"]
  },
  "defekt_analizi": {
    "gercek_delik_sayisi": "sayı",
    "gercek_budak_sayisi": "sayı", 
    "sam2_segment_sayisi": "sayı",
    "uyumsuzluk_var_mi": true/false
  },
  "kalite_degerlendirmesi": {
    "segmentasyon_basarisi": "yüzde",
    "guvenilirlik": "yüksek/orta/düşük",
    "oneriler": ["iyileştirme önerileri"]
  },
  "sonuc": "SAM2 doğru çalışmış mı yoksa yanlış pozitif mi üretmiş?"
}"""

def run_improved_analysis(image_path: str):
    """İyileştirilmiş analiz pipeline'ı çalıştırır"""
    
    print("🚀 İYİLEŞTİRİLMİŞ SAM2 + GOOGLE AI ANALİZİ")
    print("=" * 60)
    
    # Defekt odaklı prompt'ları oluştur
    improved_prompts = create_defect_focused_prompts(image_path)
    
    print("🎯 Defekt odaklı prompt'lar oluşturuldu:")
    for i, prompt in enumerate(improved_prompts["prompts"], 1):
        points = prompt["points"]
        print(f"   Nesne {i}: {len(points)} nokta")
        for j, point in enumerate(points):
            label_text = "Pozitif" if point["label"] else "Negatif"
            print(f"     {j+1}. ({point['x']}, {point['y']}) - {label_text}")
    
    # Pipeline'ı çalıştır
    pipeline = SAM2GoogleAIPipeline()
    
    # Özel analiz prompt'u ile çalıştır
    improved_analyzer = pipeline.analyzer
    improved_analyzer.create_analysis_prompt = lambda x: create_improved_ai_prompt()
    
    results = pipeline.run_full_pipeline(
        image_path=image_path,
        analysis_type="wood_inspection",
        custom_prompts=improved_prompts,
        output_prefix="improved_sam2_analysis"
    )
    
    print("\n" + "="*60)
    print("📊 İYİLEŞTİRİLMİŞ ANALİZ SONUÇLARI")
    print("="*60)
    
    if results["success"]:
        ai_analysis = results.get("ai_analysis", {}).get("analysis", {})
        
        # Segmentasyon doğruluğu
        seg_accuracy = ai_analysis.get("segmentasyon_dogrulugu", {})
        if seg_accuracy:
            print("🎯 SEGMENTASYON DOĞRULUĞU:")
            print(f"   ✅ Doğru Tespitler: {seg_accuracy.get('dogru_tespitler', [])}")
            print(f"   ❌ Yanlış Pozitifler: {seg_accuracy.get('yanlis_pozitifler', [])}")
            print(f"   ⚠️ Eksik Tespitler: {seg_accuracy.get('eksik_tespitler', [])}")
        
        # Defekt analizi
        defect_analysis = ai_analysis.get("defekt_analizi", {})
        if defect_analysis:
            print("\n🔍 DEFEKT ANALİZİ:")
            print(f"   Gerçek Delik Sayısı: {defect_analysis.get('gercek_delik_sayisi', 'N/A')}")
            print(f"   Gerçek Budak Sayısı: {defect_analysis.get('gercek_budak_sayisi', 'N/A')}")
            print(f"   SAM2 Segment Sayısı: {defect_analysis.get('sam2_segment_sayisi', 'N/A')}")
            
            mismatch = defect_analysis.get('uyumsuzluk_var_mi', False)
            mismatch_text = "🔴 EVET - SAM2 hatalı!" if mismatch else "🟢 Hayır - SAM2 doğru"
            print(f"   Uyumsuzluk: {mismatch_text}")
        
        # Kalite değerlendirmesi
        quality = ai_analysis.get("kalite_degerlendirmesi", {})
        if quality:
            print("\n📊 KALİTE DEĞERLENDİRMESİ:")
            print(f"   Başarı Oranı: {quality.get('segmentasyon_basarisi', 'N/A')}")
            print(f"   Güvenilirlik: {quality.get('guvenilirlik', 'N/A')}")
            
            recommendations = quality.get('oneriler', [])
            if recommendations:
                print("   💡 Öneriler:")
                for rec in recommendations:
                    print(f"     • {rec}")
        
        # Sonuç
        result = ai_analysis.get("sonuc", "")
        if result:
            print(f"\n🎯 SONUÇ: {result}")
    
    else:
        print(f"❌ Analiz başarısız: {results.get('error')}")
    
    return results

if __name__ == "__main__":
    image_path = "dataset/wood/test/hole/000.png"
    run_improved_analysis(image_path)