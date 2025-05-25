#!/usr/bin/env python3
"""
Report Interpreter - Analiz raporlarını kullanıcı dostu formatta açıklar
SAM2 + Google AI analiz sonuçlarını kolay anlaşılır şekilde sunar
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, Any

class ReportInterpreter:
    """Analiz raporlarını yorumlayan sınıf"""
    
    def __init__(self):
        self.severity_colors = {
            "Yok": "🟢",
            "Hafif": "🟡", 
            "Orta": "🟠",
            "Şiddetli": "🔴",
            "Kritik": "🚨"
        }
        
        self.grade_descriptions = {
            "A": "Mükemmel Kalite - Yüksek değerli uygulamalar için ideal",
            "B": "İyi Kalite - Genel amaçlı kullanım için uygun", 
            "C": "Orta Kalite - Düşük görünürlük alanları için",
            "D": "Düşük Kalite - Yapısal olmayan uygulamalar"
        }
    
    def load_report(self, report_path: str) -> Dict:
        """JSON raporu yükler"""
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ Hata: Rapor dosyası bulunamadı: {report_path}")
            return None
        except json.JSONDecodeError:
            print(f"❌ Hata: Geçersiz JSON formatı: {report_path}")
            return None
    
    def print_header(self, report: Dict):
        """Rapor başlığını yazdırır"""
        summary = report.get("pipeline_summary", {})
        
        print("\n" + "="*60)
        print("🔍 GÖRÜNTÜ ANALİZ RAPORU")
        print("="*60)
        
        print(f"📅 Tarih: {summary.get('timestamp', 'Bilinmiyor')}")
        print(f"📁 Dosya: {summary.get('input_image', 'Bilinmiyor')}")
        print(f"🔬 Analiz Türü: {summary.get('analysis_type', 'Bilinmiyor').replace('_', ' ').title()}")
        
        success = summary.get('success', False)
        status_icon = "✅" if success else "❌"
        status_text = "BAŞARILI" if success else "BAŞARISIZ"
        print(f"📊 Durum: {status_icon} {status_text}")
        print("-" * 60)
    
    def interpret_sam2_results(self, sam2_data: Dict):
        """SAM2 segmentasyon sonuçlarını yorumlar"""
        print("\n🎯 SAM2 SEGMENTASYON SONUÇLARI")
        print("-" * 40)
        
        if sam2_data.get("success"):
            print("✅ Segmentasyon başarıyla tamamlandı")
            
            output_dir = sam2_data.get("output_dir", "")
            files = sam2_data.get("files", [])
            
            print(f"📂 Çıktı Dizini: {output_dir}")
            print(f"📄 Oluşturulan Dosyalar: {len(files)} adet")
            
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    print(f"   🖼️ {file}")
                else:
                    print(f"   📄 {file}")
        else:
            error = sam2_data.get("error", "Bilinmeyen hata")
            print(f"❌ Segmentasyon başarısız: {error}")
    
    def interpret_wood_inspection(self, analysis: Dict):
        """Ahşap inceleme sonuçlarını yorumlar"""
        print("\n🌳 AHŞAP KALİTE ANALİZİ")
        print("-" * 40)
        
        # Defekt analizi
        defects = analysis.get("ahşap_defektleri", {})
        if defects:
            print("\n🔍 Tespit Edilen Defektler:")
            
            for defect_name, defect_data in defects.items():
                if isinstance(defect_data, dict) and defect_data.get("varlığı"):
                    severity = defect_data.get("şiddet", "Bilinmiyor")
                    icon = self.severity_colors.get(severity, "⚪")
                    
                    defect_tr = {
                        "budak": "Budak",
                        "çatlak": "Çatlak", 
                        "deformasyon": "Deformasyon",
                        "renk_değişikliği": "Renk Değişikliği",
                        "çürük": "Çürük"
                    }
                    
                    defect_display = defect_tr.get(defect_name, defect_name.title())
                    print(f"   {icon} {defect_display}: {severity}")
                    
                    description = defect_data.get("açıklama", "")
                    if description:
                        print(f"      💬 {description}")
        
        # Kalite değerlendirmesi
        quality = analysis.get("kalite_değerlendirmesi", {})
        if quality:
            print("\n📊 Kalite Değerlendirmesi:")
            
            grade = quality.get("ahşap_sınıfı", "Bilinmiyor")
            grade_desc = self.grade_descriptions.get(grade, "Açıklama mevcut değil")
            print(f"   🏆 Ahşap Sınıfı: Grade {grade}")
            print(f"      📝 {grade_desc}")
            
            usage = quality.get("kullanım_alanı_uygunluğu", "")
            if usage:
                print(f"   🏗️ Kullanım Alanı:")
                print(f"      📝 {usage}")
            
            recommendations = quality.get("işleme_önerileri", "")
            if recommendations:
                print(f"   💡 İşleme Önerileri:")
                print(f"      📝 {recommendations}")
    
    def interpret_anomaly_detection(self, analysis: Dict):
        """Anomali tespit sonuçlarını yorumlar"""
        print("\n🚨 ANOMALI TESPİT ANALİZİ")
        print("-" * 40)
        
        # Anomali durumu
        anomaly_detected = analysis.get("anomaly_detected", False)
        if anomaly_detected:
            print("🔴 ANOMALI TESPİT EDİLDİ!")
        else:
            print("🟢 Normal - Anomali tespit edilmedi")
        
        # Ciddiyet skoru
        severity_score = analysis.get("severity_score", 0)
        if severity_score > 0:
            severity_bar = "█" * min(severity_score, 10) + "░" * (10 - min(severity_score, 10))
            
            if severity_score <= 3:
                risk_level = "🟢 Düşük Risk"
            elif severity_score <= 6:
                risk_level = "🟡 Orta Risk"
            elif severity_score <= 8:
                risk_level = "🟠 Yüksek Risk"
            else:
                risk_level = "🔴 Kritik Risk"
            
            print(f"\n📊 Ciddiyet Skoru: {severity_score}/10")
            print(f"    [{severity_bar}] {risk_level}")
        
        # Segmentasyon kalitesi
        seg_quality = analysis.get("segmentation_quality", "unknown")
        quality_icons = {"iyi": "🟢", "orta": "🟡", "kötü": "🔴", "unknown": "⚪"}
        quality_icon = quality_icons.get(seg_quality.lower(), "⚪")
        print(f"\n🎯 Segmentasyon Kalitesi: {quality_icon} {seg_quality.title()}")
        
        # Problem alanları
        problem_areas = analysis.get("problem_areas", [])
        if problem_areas:
            print(f"\n⚠️ Problem Alanları:")
            for i, area in enumerate(problem_areas, 1):
                print(f"   {i}. {area}")
        
        # Acil dikkat
        urgent = analysis.get("urgent_attention", False)
        if urgent:
            print(f"\n🚨 ACİL DİKKAT GEREKTİRİR!")
        
        # Öneriler
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Öneriler:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        # Detaylı analiz
        detailed = analysis.get("detailed_analysis", "")
        if detailed:
            print(f"\n📝 Detaylı Analiz:")
            print(f"   {detailed}")
    
    def interpret_defect_analysis(self, analysis: Dict):
        """Defekt analiz sonuçlarını yorumlar"""
        print("\n🔧 ENDÜSTRİYEL DEFEKT ANALİZİ")
        print("-" * 40)
        
        # Defekt türleri kontrol et
        defect_types = ["crack", "hole", "stain", "deformation", "surface_roughness"]
        
        for defect in defect_types:
            if defect in analysis:
                defect_data = analysis[defect]
                if isinstance(defect_data, dict) and defect_data.get("detected", False):
                    severity = defect_data.get("severity", "Unknown")
                    icon = self.severity_colors.get(severity, "⚪")
                    
                    defect_names = {
                        "crack": "Çatlak",
                        "hole": "Delik", 
                        "stain": "Leke",
                        "deformation": "Deformasyon",
                        "surface_roughness": "Yüzey Pürüzlülüğü"
                    }
                    
                    defect_name = defect_names.get(defect, defect.title())
                    print(f"   {icon} {defect_name}: {severity}")
        
        # Kalite sınıfı
        quality_class = analysis.get("quality_class", "")
        if quality_class:
            print(f"\n🏆 Kalite Sınıfı: {quality_class}")
        
        # Ret durumu
        rejection = analysis.get("should_reject", False)
        if rejection:
            print("🚫 ÖNERİ: Ürün ret edilmeli")
        else:
            print("✅ ÖNERİ: Ürün kabul edilebilir")
        
        # Onarım durumu
        repairable = analysis.get("repairable", False)
        if repairable:
            print("🔧 Onarım mümkün")
        else:
            print("❌ Onarım önerilmez")
    
    def interpret_general_analysis(self, analysis: Dict):
        """Genel analiz sonuçlarını yorumlar"""
        print("\n📋 GENEL ANALİZ SONUÇLARI")
        print("-" * 40)
        
        # Raw text varsa göster
        raw_text = analysis.get("raw_text", "")
        if raw_text:
            print(f"📝 Analiz Sonucu:")
            print(f"   {raw_text}")
        else:
            # Diğer alanları kontrol et
            for key, value in analysis.items():
                if key != "raw_text":
                    print(f"   {key}: {value}")
    
    def interpret_report(self, report_path: str):
        """Ana rapor yorumlama fonksiyonu"""
        report = self.load_report(report_path)
        if not report:
            return
        
        # Başlık
        self.print_header(report)
        
        # SAM2 sonuçları
        sam2_data = report.get("sam2_segmentation", {})
        if sam2_data:
            self.interpret_sam2_results(sam2_data)
        
        # AI analiz sonuçları
        ai_analysis = report.get("ai_analysis", {})
        if ai_analysis.get("success"):
            analysis_data = ai_analysis.get("analysis", {})
            analysis_type = report.get("pipeline_summary", {}).get("analysis_type", "")
            
            # Analiz türüne göre yorumla
            if analysis_type == "wood_inspection":
                self.interpret_wood_inspection(analysis_data)
            elif analysis_type == "anomaly_detection":
                self.interpret_anomaly_detection(analysis_data)
            elif analysis_type == "defect_analysis":
                self.interpret_defect_analysis(analysis_data)
            else:
                self.interpret_general_analysis(analysis_data)
        else:
            error = ai_analysis.get("error", "Bilinmeyen hata")
            print(f"\n❌ AI Analizi Başarısız: {error}")
        
        # Hata mesajı
        error = report.get("error")
        if error:
            print(f"\n🚨 HATA: {error}")
        
        # Summary bilgileri
        summary = report.get("summary", {})
        if summary:
            print(f"\n📊 ÖZET BİLGİLER")
            print("-" * 40)
            
            for key, value in summary.items():
                key_tr = {
                    "anomaly_detected": "Anomali Tespit",
                    "severity_score": "Ciddiyet Skoru",
                    "segmentation_quality": "Segmentasyon Kalitesi",
                    "urgent_attention": "Acil Dikkat",
                    "recommendations": "Öneri Sayısı"
                }
                
                display_key = key_tr.get(key, key.replace("_", " ").title())
                
                if isinstance(value, bool):
                    display_value = "Evet" if value else "Hayır"
                elif isinstance(value, list):
                    display_value = f"{len(value)} adet"
                else:
                    display_value = str(value)
                
                print(f"   {display_key}: {display_value}")
        
        print("\n" + "="*60)
        print("📄 Rapor yorumlama tamamlandı!")
        print("="*60)

def main():
    """Ana fonksiyon"""
    
    print("📊 SAM2 + Google AI Rapor Yorumlayıcı")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("Kullanım: python report_interpreter.py <report_file.json>")
        print("\nÖrnek:")
        print("  python report_interpreter.py sam2_ai_analysis_report_1748170015.json")
        
        # Mevcut rapor dosyalarını listele
        current_dir = "."
        json_files = [f for f in os.listdir(current_dir) if f.endswith('.json') and 'report' in f]
        
        if json_files:
            print(f"\nMevcut rapor dosyaları:")
            for i, file in enumerate(json_files, 1):
                print(f"  {i}. {file}")
        
        return
    
    report_file = sys.argv[1]
    
    if not os.path.exists(report_file):
        print(f"❌ Hata: Rapor dosyası bulunamadı: {report_file}")
        return
    
    try:
        interpreter = ReportInterpreter()
        interpreter.interpret_report(report_file)
        
    except KeyboardInterrupt:
        print("\n⚠️ İşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")

if __name__ == "__main__":
    main()