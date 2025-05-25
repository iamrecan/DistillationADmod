#!/usr/bin/env python3
"""
LLM Image Analysis Script
Google AI ile segmentasyon sonuçlarını analiz eder ve anomali tespiti yapar
"""

import os
import sys
import json
import base64
import requests
import cv2
import numpy as np
from PIL import Image
import time
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv

# .env dosyasından çevre değişkenlerini yükle
load_dotenv()

class GoogleAIImageAnalyzer:
    """Google AI API ile görüntü analizi yapan sınıf"""
    
    def __init__(self):
        """Initialize the Google AI analyzer"""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY çevre değişkeni bulunamadı! .env dosyasını kontrol edin.")
        
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        self.headers = {
            "Content-Type": "application/json"
        }
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Görüntüyü base64 formatına çevirir"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Görüntü kodlama hatası: {e}")
            return None
    
    def create_analysis_prompt(self, analysis_type: str) -> str:
        """Analiz türüne göre prompt oluşturur"""
        
        if analysis_type == "wood_inspection":
            return """Bu görüntü SAM2 ile segmente edilmiş bir ahşap yüzey parçasıdır.

🎯 ÖNEMLİ: Bu segmente edilen bölge, üretim kalite kontrolünde ANOMALI olarak işaretlenmiştir.
Sizin göreviniz bu anomali bölgesinin NE TÜR BİR PROBLEM olduğunu belirlemektir.

ANOMALI TÜRLERİ:
1. **DELIK (Hole)**: Açık, koyu renkli, düzensiz şekilli boşluklar
2. **BUDAK (Knot)**: Oval, gömülü, doğal ama kaliteyi düşüren nodüller  
3. **ÇATLAK (Crack)**: İnce çizgi şeklindeki yarıklar
4. **RENK DEĞİŞİKLİĞİ**: Leke, pas, küf benzeri renklenme
5. **DEFORMASYON**: Şekil bozukluğu, çarpıklık
6. **YÜZEY PÜRÜZLÜĞÜ**: Düzensiz doku, çapaklar

DİKKAT: Bu bölge zaten anomali olarak tespit edilmiştir, sizin göreviniz türünü belirlemek!

🎯 RECOMMENDED ACTIONS (Ne yapılmalı?):
- **KABUL_ET**: Ürün kabul edilebilir, sevkiyat yapılabilir
- **ONAR**: Küçük müdahale ile düzeltilebilir (dolgu, zımpara, etc.)
- **REDDET**: Ürün kalite standartlarını karşılamıyor, reddedilmeli
- **MANUEL_KONTROL**: Uzman kontrolüne ihtiyaç var
- **SINIFLANDIR**: Düşük kalite sınıfına dahil et

JSON formatında yanıt verin:
{
  "anomali_turu": "tespit edilen anomali türü",
  "detayli_aciklama": "anomalinin görsel özellikleri",
  "siddet_seviyesi": "Hafif/Orta/Şiddetli",
  "uretim_etkisi": "bu anomalinin üretim kalitesine etkisi",
  "cozum_onerisi": "ne yapılmalı (onar/reddet/kabul et)",
  "recommended_action": "KABUL_ET/ONAR/REDDET/MANUEL_KONTROL/SINIFLANDIR",
  "action_details": "önerilen aksiyonun detayları",
  "estimated_cost": "tahmini maliyet etkisi (Düşük/Orta/Yüksek)",
  "guven_skoru": "0-100 arası tanımlama güveni"
}"""
        
        elif analysis_type == "anomaly_detection":
            return """Bu SAM2 segmentasyonu ile işaretlenmiş bölgeyi analiz edin.

🎯 GÖREV: Bu bölgede hangi tür anomali var?

ANOMALI KATEGORİLERİ:
- Yapısal defekt (çatlak, kırık, delik)
- Yüzey defekti (çizik, leke, renk değişikliği)  
- Deformasyon (şekil bozukluğu, çarpıklık)
- Kirlilik (yabancı madde, pas, küf)

JSON formatında yanıt verin:
{
  "anomaly_detected": true,
  "anomaly_type": "tespit edilen anomali türü", 
  "severity_score": "1-10 arası ciddiyet skoru",
  "detailed_analysis": "detaylı açıklama",
  "urgent_attention": "acil müdahale gerekli mi?",
  "recommendations": ["yapılması gereken işlemler"]
}"""
        
        elif analysis_type == "defect_analysis":
            return """Bu segmente edilmiş bölgede ENDÜSTRİYEL DEFEKT analizi yapın.

🎯 Bu bölge üretim kalite kontrolünde anomali olarak işaretlenmiştir.

DEFEKT TÜRLERİ:
- **CRACK**: Çatlak, kırık, yarık
- **HOLE**: Delik, boşluk, kavite
- **STAIN**: Leke, renk değişikliği, kirlilik
- **DEFORMATION**: Şekil bozukluğu, çarpıklık
- **SURFACE_ROUGHNESS**: Yüzey pürüzlülüğü, doku problemi

JSON formatında yanıt verin:
{
  "defect_type": "tespit edilen defekt türü",
  "severity": "Hafif/Orta/Şiddetli/Kritik",
  "quality_impact": "kalite üzerindeki etkisi",
  "should_reject": "ürün reddedilmeli mi?",
  "repairable": "onarılabilir mi?", 
  "confidence": "0-100 arası güven skoru"
}"""
        
        else:
            return """Bu SAM2 segmentasyonu ile işaretlenmiş bölgeyi analiz edin ve anomali türünü belirleyin.

Bu bölge üretim kalite kontrolünde ANOMALI olarak işaretlenmiştir.
Göreviniz bu anomalinin ne tür bir problem olduğunu belirlemektir.

JSON formatında yanıt verin:
{
  "anomaly_type": "tespit edilen anomali türü",
  "description": "detaylı açıklama", 
  "severity": "problem ciddiyeti",
  "recommendation": "önerilen işlem"
}"""
    
    def analyze_image(self, image_path: str, analysis_type: str = "anomaly_detection", 
                     custom_prompt: str = None) -> Dict:
        """Görüntüyü Google AI ile analiz eder"""
        
        # Görüntüyü base64'e çevir
        base64_image = self.encode_image_to_base64(image_path)
        if not base64_image:
            return {"error": "Görüntü kodlanamadı"}
        
        # Prompt oluştur
        prompt = custom_prompt if custom_prompt else self.create_analysis_prompt(analysis_type)
        
        # API request payload oluştur
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 32,
                "topP": 1,
                "maxOutputTokens": 4096,
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        try:
            # API çağrısı yap
            url = f"{self.base_url}?key={self.api_key}"
            response = requests.post(url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                text_response = result['candidates'][0]['content']['parts'][0]['text']
                
                # JSON yanıtı parse etmeye çalış
                try:
                    # JSON kısmını bul ve parse et
                    import re
                    json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        parsed_response = json.loads(json_str)
                        return {
                            "success": True,
                            "analysis": parsed_response,
                            "raw_response": text_response,
                            "timestamp": time.time()
                        }
                    else:
                        return {
                            "success": True,
                            "analysis": {"raw_text": text_response},
                            "raw_response": text_response,
                            "timestamp": time.time()
                        }
                except json.JSONDecodeError:
                    return {
                        "success": True,
                        "analysis": {"raw_text": text_response},
                        "raw_response": text_response,
                        "timestamp": time.time()
                    }
            else:
                return {"error": "Google AI'den geçerli yanıt alınamadı"}
                
        except requests.exceptions.RequestException as e:
            return {"error": f"API isteği başarısız: {str(e)}"}
        except Exception as e:
            return {"error": f"Beklenmeyen hata: {str(e)}"}
    
    def analyze_segmented_image(self, original_image: str, segmented_image: str, 
                               analysis_type: str = "anomaly_detection") -> Dict:
        """Orijinal ve segmentasyon sonucunu karşılaştırmalı analiz eder"""
        
        # İki görüntüyü yan yana birleştir
        try:
            # Görüntüleri yükle
            img1 = cv2.imread(original_image)
            img2 = cv2.imread(segmented_image)
            
            if img1 is None or img2 is None:
                return {"error": "Görüntüler yüklenemedi"}
            
            # Görüntüleri aynı boyuta getir
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            max_h = max(h1, h2)
            
            img1_resized = cv2.resize(img1, (w1 * max_h // h1, max_h))
            img2_resized = cv2.resize(img2, (w2 * max_h // h2, max_h))
            
            # Yan yana birleştir
            combined = np.hstack([img1_resized, img2_resized])
            
            # Geçici dosya olarak kaydet
            temp_path = "temp_combined_analysis.jpg"
            cv2.imwrite(temp_path, combined)
            
            # Özel prompt oluştur
            custom_prompt = f"""Sol tarafta orijinal görüntü, sağ tarafta SAM2 segmentasyon sonucu var. 
Karşılaştırmalı analiz yapın:

1. **Segmentasyon Başarısı:**
   - Segment sınırları doğru mu?
   - Eksik veya yanlış segmentler var mı?
   - Hassasiyet ve doğruluk nasıl?

2. **Anomali Tespiti:**
   - Segmentasyon hangi anomalileri yakaladı?
   - Gözden kaçan problemler var mı?
   - False positive/negative durumlar?

3. **Karşılaştırmalı Değerlendirme:**
   - Segmentasyon kalitesi genel olarak nasıl?
   - İyileştirme önerileri?

{self.create_analysis_prompt(analysis_type)}"""
            
            # Analiz yap
            result = self.analyze_image(temp_path, analysis_type, custom_prompt)
            
            # Geçici dosyayı temizle
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return result
            
        except Exception as e:
            return {"error": f"Karşılaştırmalı analiz hatası: {str(e)}"}

def create_analysis_report(analysis_results: Dict, output_path: str = "analysis_report.json"):
    """Analiz sonuçlarından rapor oluşturur"""
    
    if not analysis_results.get("success"):
        print(f"Analiz başarısız: {analysis_results.get('error', 'Bilinmeyen hata')}")
        return
    
    # Rapor oluştur
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_results": analysis_results,
        "summary": {}
    }
    
    # Analiz sonuçlarından özet çıkar
    analysis = analysis_results.get("analysis", {})
    if isinstance(analysis, dict):
        report["summary"] = {
            "anomaly_detected": analysis.get("anomaly_detected", False),
            "severity_score": analysis.get("severity_score", 0),
            "segmentation_quality": analysis.get("segmentation_quality", "unknown"),
            "urgent_attention": analysis.get("urgent_attention", False)
        }
    
    # JSON olarak kaydet
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Analiz raporu kaydedildi: {output_path}")
    return report

def main():
    """Ana fonksiyon - komut satırından kullanım"""
    
    if len(sys.argv) < 3:
        print("Kullanım: python llm_image_analysis.py <image_path> <analysis_type> [output_report]")
        print("Analysis types: anomaly_detection, defect_analysis, wood_inspection")
        print("Örnek: python llm_image_analysis.py image.jpg anomaly_detection report.json")
        return
    
    image_path = sys.argv[1]
    analysis_type = sys.argv[2] if len(sys.argv) > 2 else "anomaly_detection"
    output_report = sys.argv[3] if len(sys.argv) > 3 else "analysis_report.json"
    
    if not os.path.exists(image_path):
        print(f"Hata: Görüntü dosyası bulunamadı: {image_path}")
        return
    
    try:
        # Analyzer'ı başlat
        analyzer = GoogleAIImageAnalyzer()
        print(f"Google AI ile görüntü analizi başlatılıyor...")
        print(f"Görüntü: {image_path}")
        print(f"Analiz türü: {analysis_type}")
        
        # Analiz yap
        results = analyzer.analyze_image(image_path, analysis_type)
        
        if results.get("success"):
            print("✓ Analiz başarıyla tamamlandı!")
            
            # Sonuçları göster
            analysis = results.get("analysis", {})
            if isinstance(analysis, dict):
                print("\n" + "="*50)
                print("ANALIZ SONUÇLARI")
                print("="*50)
                
                for key, value in analysis.items():
                    if key != "detailed_analysis":
                        print(f"{key}: {value}")
                
                if "detailed_analysis" in analysis:
                    print(f"\nDetaylı Analiz:\n{analysis['detailed_analysis']}")
            else:
                print(f"Analiz sonucu: {analysis}")
        else:
            print(f"❌ Analiz başarısız: {results.get('error')}")
        
        # Rapor oluştur
        create_analysis_report(results, output_report)
        
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    main()