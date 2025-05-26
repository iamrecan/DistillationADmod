#!/usr/bin/env python3
"""
Integrated SAM2 + Google AI Analysis Pipeline
SAM2 ile segmentasyon yapıp Google AI ile analiz eden entegre sistem
"""

import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Custom imports
from llm_image_analysis import GoogleAIImageAnalyzer, create_analysis_report
from analysis_config import AnalysisConfig

# .env dosyasından çevre değişkenlerini yükle
load_dotenv()

class SAM2GoogleAIPipeline:
    """SAM2 segmentasyon + Google AI analiz pipeline'ı"""
    
    def __init__(self):
        """Initialize the pipeline"""
        self.analyzer = GoogleAIImageAnalyzer()
        self.temp_dir = AnalysisConfig.get_temp_analysis_path("")
        self.ensure_temp_dir()
    
    def ensure_temp_dir(self):
        """Geçici dizini oluştur"""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def run_sam2_segmentation(self, image_path: str, prompts: Dict, output_dir: str) -> Dict:
        """SAM2 segmentasyonu çalıştır"""
        
        try:
            print("🔄 SAM2 segmentasyonu başlatılıyor...")
            
            # Prompt'u JSON string'e çevir
            prompt_json = json.dumps(prompts)
            
            # send_requestSam2.py'yi çalıştır
            cmd = [
                "python", "send_requestSam2.py",
                prompt_json,
                image_path,
                output_dir
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✓ SAM2 segmentasyonu tamamlandı!")
                
                # Çıktı dosyalarını kontrol et
                if os.path.exists(output_dir):
                    files = os.listdir(output_dir)
                    return {
                        "success": True,
                        "output_dir": output_dir,
                        "files": files,
                        "stdout": result.stdout
                    }
                else:
                    return {"success": False, "error": "Çıktı dizini oluşturulamadı"}
            else:
                return {
                    "success": False,
                    "error": f"SAM2 hatası: {result.stderr}",
                    "stdout": result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "SAM2 zaman aşımına uğradı"}
        except Exception as e:
            return {"success": False, "error": f"SAM2 çalıştırma hatası: {str(e)}"}
    
    def find_output_images(self, output_dir: str) -> Dict[str, str]:
        """Çıktı dizinindeki görüntüleri bul"""
        
        image_files = {}
        if not os.path.exists(output_dir):
            return image_files
        
        # Yaygın görüntü formatlarını ara
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file.lower())
                if ext in image_extensions:
                    # Dosya türünü belirle
                    if 'mask' in file.lower() or 'segment' in file.lower():
                        image_files['segmented'] = file_path
                    elif 'output' in file.lower() or 'result' in file.lower():
                        image_files['result'] = file_path
                    else:
                        image_files['other'] = file_path
        
        return image_files
    
    def analyze_with_google_ai(self, image_path: str, analysis_type: str = "anomaly_detection",
                              segmented_image: str = None) -> Dict:
        """Google AI ile analiz yap"""
        
        print("🔄 Google AI analizi başlatılıyor...")
        
        try:
            if segmented_image and os.path.exists(segmented_image):
                # Karşılaştırmalı analiz
                result = self.analyzer.analyze_segmented_image(
                    image_path, segmented_image, analysis_type
                )
            else:
                # Tek görüntü analizi
                result = self.analyzer.analyze_image(image_path, analysis_type)
            
            if result.get("success"):
                print("✓ Google AI analizi tamamlandı!")
            else:
                print(f"❌ Google AI analizi başarısız: {result.get('error')}")
            
            return result
            
        except Exception as e:
            return {"error": f"Google AI analiz hatası: {str(e)}"}
    
    def create_default_prompts(self, image_path: str) -> Dict:
        """Varsayılan SAM2 prompt'ları oluştur"""
        
        # Görüntü boyutlarını al
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Görüntü yüklenemedi")
            
            h, w = img.shape[:2]
            
            # Görüntünün merkezine yakın noktalar seç
            center_x, center_y = w // 2, h // 2
            
            prompts = {
                "prompts": [
                    {
                        "type": "points",
                        "object_id": 1,
                        "frame_index": 0,
                        "points": [
                            {"x": center_x, "y": center_y, "label": True},
                            {"x": center_x - 50, "y": center_y - 50, "label": True},
                            {"x": center_x + 50, "y": center_y + 50, "label": True}
                        ]
                    }
                ]
            }
            
            return prompts
            
        except Exception as e:
            print(f"Varsayılan prompt oluşturma hatası: {e}")
            # Fallback prompt
            return {
                "prompts": [
                    {
                        "type": "points",
                        "object_id": 1,
                        "frame_index": 0,
                        "points": [
                            {"x": 200, "y": 200, "label": True}
                        ]
                    }
                ]
            }
    
    def run_full_pipeline(self, image_path: str, analysis_type: str = "anomaly_detection",
                         custom_prompts: Dict = None, output_prefix: str = "analysis") -> Dict:
        """Tam pipeline'ı çalıştır: SAM2 + Google AI"""
        
        print("🚀 SAM2 + Google AI Pipeline başlatılıyor...")
        print(f"📁 Görüntü: {image_path}")
        print(f"🔍 Analiz türü: {analysis_type}")
        
        # Çıktı dizinlerini hazırla
        timestamp = int(time.time())
        sam2_output_dir = AnalysisConfig.get_sam2_output_dir(timestamp)
        
        pipeline_results = {
            "timestamp": timestamp,
            "input_image": image_path,
            "analysis_type": analysis_type,
            "sam2_results": {},
            "ai_analysis": {},
            "success": False
        }
        
        try:
            # 1. SAM2 Segmentasyon
            prompts = custom_prompts if custom_prompts else self.create_default_prompts(image_path)
            sam2_result = self.run_sam2_segmentation(image_path, prompts, sam2_output_dir)
            pipeline_results["sam2_results"] = sam2_result
            
            if not sam2_result.get("success"):
                pipeline_results["error"] = f"SAM2 başarısız: {sam2_result.get('error')}"
                return pipeline_results
            
            # 2. Çıktı görüntülerini bul
            output_images = self.find_output_images(sam2_output_dir)
            pipeline_results["output_images"] = output_images
            
            # 3. Google AI Analizi
            segmented_image = output_images.get('segmented') or output_images.get('result')
            ai_result = self.analyze_with_google_ai(
                image_path, analysis_type, segmented_image
            )
            pipeline_results["ai_analysis"] = ai_result
            
            # 4. Başarı durumu
            if ai_result.get("success"):
                pipeline_results["success"] = True
                print("🎉 Pipeline başarıyla tamamlandı!")
            else:
                pipeline_results["error"] = f"AI analizi başarısız: {ai_result.get('error')}"
            
            # 5. Rapor oluştur
            report_path = AnalysisConfig.get_sam2_report_path(timestamp)
            self.create_pipeline_report(pipeline_results, report_path)
            pipeline_results["report_path"] = report_path
            
            return pipeline_results
            
        except Exception as e:
            pipeline_results["error"] = f"Pipeline hatası: {str(e)}"
            return pipeline_results
    
    def create_pipeline_report(self, results: Dict, output_path: str):
        """Pipeline raporu oluştur"""
        
        # Detaylı rapor hazırla
        report = {
            "pipeline_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(results["timestamp"])),
                "success": results["success"],
                "input_image": results["input_image"],
                "analysis_type": results["analysis_type"]
            },
            "sam2_segmentation": results.get("sam2_results", {}),
            "ai_analysis": results.get("ai_analysis", {}),
            "output_files": results.get("output_images", {}),
            "error": results.get("error")
        }
        
        # Özet bilgiler - YENİ ve İYİLEŞTİRİLMİŞ
        summary = {
            "anomaly_detected": False,
            "severity_score": 0,
            "segmentation_quality": "unknown",
            "urgent_attention": False,
            "recommendations": []
        }
        
        if results.get("success") and "ai_analysis" in results:
            ai_analysis = results["ai_analysis"].get("analysis", {})
            
            # YENİ JSON formatını kontrol et (wood_inspection türü)
            if isinstance(ai_analysis, dict):
                # Anomali türü kontrolü - yeni format
                anomali_turu = ai_analysis.get("anomali_turu", "")
                if anomali_turu:
                    summary["anomaly_detected"] = True
                    
                    # Ciddiyet seviyesini skorla
                    siddet = ai_analysis.get("siddet_seviyesi", "").lower()
                    if "hafif" in siddet:
                        summary["severity_score"] = 3
                    elif "orta" in siddet:
                        summary["severity_score"] = 6
                    elif "şiddetli" in siddet or "kritik" in siddet:
                        summary["severity_score"] = 9
                        summary["urgent_attention"] = True
                    
                    # Segmentasyon kalitesini güven skorundan çıkar
                    guven_skoru = ai_analysis.get("guven_skoru", "0")
                    try:
                        guven = int(str(guven_skoru))
                        if guven >= 90:
                            summary["segmentation_quality"] = "yüksek"
                        elif guven >= 70:
                            summary["segmentation_quality"] = "orta" 
                        else:
                            summary["segmentation_quality"] = "düşük"
                    except:
                        pass
                    
                    # Çözüm önerisini recommendations'a ekle
                    cozum = ai_analysis.get("cozum_onerisi", "")
                    if cozum:
                        summary["recommendations"] = [cozum]
                    
                    # YENİ: Recommended action ve detayları ekle
                    recommended_action = ai_analysis.get("recommended_action", "")
                    action_details = ai_analysis.get("action_details", "")
                    estimated_cost = ai_analysis.get("estimated_cost", "")
                    
                    if recommended_action:
                        summary["recommended_action"] = recommended_action
                        summary["action_details"] = action_details
                        summary["estimated_cost"] = estimated_cost
                        
                        # Acil durumları işaretle
                        if recommended_action in ["REDDET", "MANUEL_KONTROL"]:
                            summary["urgent_attention"] = True
                        elif recommended_action == "ONAR" and estimated_cost in ["Orta", "Yüksek"]:
                            summary["urgent_attention"] = True
                    
                    print(f"🎯 YENİ FORMAT TESPİT EDİLDİ:")
                    print(f"   Anomali Türü: {anomali_turu}")
                    print(f"   Ciddiyet: {siddet} -> Skor: {summary['severity_score']}")
                    print(f"   Güven: {guven_skoru}% -> Kalite: {summary['segmentation_quality']}")
                    print(f"   Önerilen Aksiyon: {recommended_action}")
                    print(f"   Maliyet Etkisi: {estimated_cost}")
                
                # ESKİ JSON formatını kontrol et (diğer türler)
                elif ai_analysis.get("anomaly_detected") or ai_analysis.get("defect_type"):
                    summary.update({
                        "anomaly_detected": ai_analysis.get("anomaly_detected", False),
                        "severity_score": ai_analysis.get("severity_score", 0),
                        "segmentation_quality": ai_analysis.get("segmentation_quality", "unknown"),
                        "urgent_attention": ai_analysis.get("urgent_attention", False),
                        "recommendations": ai_analysis.get("recommendations", [])
                    })
        
        report["summary"] = summary
        
        # Raporu kaydet
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Pipeline raporu kaydedildi: {output_path}")
        
        # Debug: Özet bilgilerini göster
        print(f"🔍 Özet bilgiler:")
        print(f"   Anomali: {'EVET' if summary['anomaly_detected'] else 'HAYIR'}")
        print(f"   Ciddiyet: {summary['severity_score']}/10")
        print(f"   Kalite: {summary['segmentation_quality']}")
        print(f"   Acil: {'EVET' if summary['urgent_attention'] else 'HAYIR'}")
    
    def cleanup_temp_files(self):
        """Geçici dosyaları temizle"""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print("🧹 Geçici dosyalar temizlendi")
        except Exception as e:
            print(f"Temizlik hatası: {e}")

def create_custom_prompts_interactive():
    """Kullanıcıdan interaktif olarak prompt'ları al"""
    
    print("\n📝 SAM2 için custom prompt'ları oluşturun:")
    print("Boş bırakırsanız varsayılan prompt'lar kullanılacak.")
    
    prompts = {"prompts": []}
    
    try:
        object_count = input("Kaç nesne segmentasyon yapmak istiyorsunuz? (varsayılan: 1): ")
        object_count = int(object_count) if object_count.strip() else 1
        
        for i in range(object_count):
            print(f"\n--- Nesne {i+1} ---")
            points = []
            
            point_count = input(f"Nesne {i+1} için kaç nokta eklemek istiyorsunuz? (varsayılan: 1): ")
            point_count = int(point_count) if point_count.strip() else 1
            
            for j in range(point_count):
                print(f"Nokta {j+1}:")
                x = input("  X koordinatı: ")
                y = input("  Y koordinatı: ")
                label = input("  Pozitif nokta mı? (y/n, varsayılan: y): ").lower()
                
                if x.strip() and y.strip():
                    points.append({
                        "x": int(x),
                        "y": int(y),
                        "label": label != 'n'
                    })
            
            if points:
                prompts["prompts"].append({
                    "type": "points",
                    "object_id": i + 1,
                    "frame_index": 0,
                    "points": points
                })
        
        return prompts if prompts["prompts"] else None
        
    except (ValueError, KeyboardInterrupt):
        print("Varsayılan prompt'lar kullanılacak.")
        return None

def main():
    """Ana fonksiyon"""
    
    print("🎯 SAM2 + Google AI Anomali Tespit Sistemi")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Kullanım: python sam2_google_ai_pipeline.py <image_path> [analysis_type] [--interactive]")
        print("Analysis types: anomaly_detection, defect_analysis, wood_inspection")
        print("--interactive: Custom prompt'ları interaktif olarak belirle")
        print("\nÖrnek:")
        print("  python sam2_google_ai_pipeline.py image.jpg anomaly_detection")
        print("  python sam2_google_ai_pipeline.py image.jpg wood_inspection --interactive")
        return
    
    image_path = sys.argv[1]
    analysis_type = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else "anomaly_detection"
    interactive = "--interactive" in sys.argv
    
    # Dosya kontrolü
    if not os.path.exists(image_path):
        print(f"❌ Hata: Görüntü dosyası bulunamadı: {image_path}")
        return
    
    try:
        # Pipeline'ı başlat
        pipeline = SAM2GoogleAIPipeline()
        
        # Custom prompt'ları al
        custom_prompts = None
        if interactive:
            custom_prompts = create_custom_prompts_interactive()
        
        # Pipeline'ı çalıştır
        results = pipeline.run_full_pipeline(
            image_path=image_path,
            analysis_type=analysis_type,
            custom_prompts=custom_prompts,
            output_prefix="sam2_ai_analysis"
        )
        
        # Sonuçları göster
        if results["success"]:
            print("\n🎉 Analiz başarıyla tamamlandı!")
            
            # AI analiz sonuçlarını göster
            ai_analysis = results.get("ai_analysis", {}).get("analysis", {})
            if isinstance(ai_analysis, dict):
                print("\n📊 ANALIZ SONUÇLARI:")
                print("-" * 30)
                
                # Ana metrikler
                if "anomaly_detected" in ai_analysis:
                    status = "🔴 ANOMALI TESPİT EDİLDİ" if ai_analysis["anomaly_detected"] else "🟢 NORMAL"
                    print(f"Durum: {status}")
                
                if "severity_score" in ai_analysis:
                    score = ai_analysis["severity_score"]
                    print(f"Ciddiyet Skoru: {score}/10")
                
                if "segmentation_quality" in ai_analysis:
                    print(f"Segmentasyon Kalitesi: {ai_analysis['segmentation_quality']}")
                
                if "urgent_attention" in ai_analysis:
                    urgent = "⚠️ ACİL DİKKAT GEREKTİRİR" if ai_analysis["urgent_attention"] else "✅ Rutin"
                    print(f"Aciliyet: {urgent}")
                
                # Öneriler
                if "recommendations" in ai_analysis and ai_analysis["recommendations"]:
                    print("\n💡 ÖNERİLER:")
                    for i, rec in enumerate(ai_analysis["recommendations"], 1):
                        print(f"  {i}. {rec}")
                
                # Detaylı analiz
                if "detailed_analysis" in ai_analysis:
                    print(f"\n📝 DETAYLI ANALİZ:\n{ai_analysis['detailed_analysis']}")
        else:
            print(f"\n❌ Analiz başarısız: {results.get('error')}")
        
        # Rapor yolu
        if "report_path" in results:
            print(f"\n📄 Detaylı rapor: {results['report_path']}")
        
        # Temizlik seçeneği
        cleanup = input("\nGeçici dosyalar silinsin mi? (y/n, varsayılan: y): ")
        if cleanup.lower() != 'n':
            pipeline.cleanup_temp_files()
    
    except KeyboardInterrupt:
        print("\n⚠️ İşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")

if __name__ == "__main__":
    main()