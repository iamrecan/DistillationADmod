#!/usr/bin/env python3
"""
Analysis Configuration - Analiz raporlarının konumlarını ve ayarlarını yönetir
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

class AnalysisConfig:
    """Analiz raporları için konfigürasyon sınıfı"""
    
    # Ana dizinler
    BASE_DIR = Path(__file__).parent
    REPORTS_DIR = BASE_DIR / "reports"
    TEMP_DIR = BASE_DIR / "temp_analysis"
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_INTERACTIVE_DIR = BASE_DIR / "results_interactive_v2"
    
    # Rapor türleri için alt dizinler
    INTEGRATED_REPORTS_DIR = REPORTS_DIR / "integrated"
    SAM2_REPORTS_DIR = REPORTS_DIR / "sam2"
    LLM_REPORTS_DIR = REPORTS_DIR / "llm"
    DEBUG_REPORTS_DIR = REPORTS_DIR / "debug"
    
    # Dosya adı formatları
    INTEGRATED_REPORT_FORMAT = "integrated_analysis_report_{timestamp}.json"
    SAM2_REPORT_FORMAT = "sam2_analysis_report_{timestamp}.json"
    LLM_REPORT_FORMAT = "llm_analysis_report_{timestamp}.json"
    DEBUG_REPORT_FORMAT = "debug_analysis_report_{timestamp}.json"
    
    # Görselleştirme formatları
    ANOMALY_VISUALIZATION_FORMAT = "anomaly_comparison_{model}_{dataset}_{timestamp}.png"
    SAM2_OUTPUT_FORMAT = "sam2_output_{timestamp}"
    
    # Temizlik ayarları
    MAX_REPORTS_PER_TYPE = 50  # Tip başına maksimum rapor sayısı
    AUTO_CLEANUP_ENABLED = True
    KEEP_DAYS = 30  # Kaç gün tutulacak
    
    @classmethod
    def initialize_directories(cls):
        """Gerekli dizinleri oluştur"""
        directories = [
            cls.REPORTS_DIR,
            cls.TEMP_DIR,
            cls.RESULTS_DIR,
            cls.RESULTS_INTERACTIVE_DIR,
            cls.INTEGRATED_REPORTS_DIR,
            cls.SAM2_REPORTS_DIR,
            cls.LLM_REPORTS_DIR,
            cls.DEBUG_REPORTS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
        print(f"📁 Analiz dizinleri hazırlandı:")
        print(f"   📊 Raporlar: {cls.REPORTS_DIR}")
        print(f"   🔧 Geçici: {cls.TEMP_DIR}")
        print(f"   📸 Sonuçlar: {cls.RESULTS_DIR}")
    
    @classmethod
    def get_integrated_report_path(cls, timestamp: Optional[int] = None) -> str:
        """Entegre rapor dosya yolunu al"""
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())
        
        filename = cls.INTEGRATED_REPORT_FORMAT.format(timestamp=timestamp)
        return str(cls.INTEGRATED_REPORTS_DIR / filename)
    
    @classmethod
    def get_sam2_report_path(cls, timestamp: Optional[int] = None) -> str:
        """SAM2 rapor dosya yolunu al"""
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())
        
        filename = cls.SAM2_REPORT_FORMAT.format(timestamp=timestamp)
        return str(cls.SAM2_REPORTS_DIR / filename)
    
    @classmethod
    def get_llm_report_path(cls, timestamp: Optional[int] = None) -> str:
        """LLM rapor dosya yolunu al"""
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())
        
        filename = cls.LLM_REPORT_FORMAT.format(timestamp=timestamp)
        return str(cls.LLM_REPORTS_DIR / filename)
    
    @classmethod
    def get_debug_report_path(cls, timestamp: Optional[int] = None) -> str:
        """Debug rapor dosya yolunu al"""
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())
        
        filename = cls.DEBUG_REPORT_FORMAT.format(timestamp=timestamp)
        return str(cls.DEBUG_REPORTS_DIR / filename)
    
    @classmethod
    def get_anomaly_visualization_path(cls, model: str, dataset: str, 
                                     timestamp: Optional[int] = None) -> str:
        """Anomali görselleştirme dosya yolunu al"""
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())
        
        filename = cls.ANOMALY_VISUALIZATION_FORMAT.format(
            model=model, dataset=dataset, timestamp=timestamp
        )
        return str(cls.RESULTS_DIR / filename)
    
    @classmethod
    def get_sam2_output_dir(cls, timestamp: Optional[int] = None) -> str:
        """SAM2 çıktı dizini yolunu al"""
        if timestamp is None:
            timestamp = int(datetime.now().timestamp())
        
        dirname = cls.SAM2_OUTPUT_FORMAT.format(timestamp=timestamp)
        output_dir = cls.TEMP_DIR / dirname
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir)
    
    @classmethod
    def get_temp_analysis_path(cls, filename: str) -> str:
        """Geçici analiz dosyası yolunu al"""
        return str(cls.TEMP_DIR / filename)
    
    @classmethod
    def cleanup_old_reports(cls):
        """Eski raporları temizle"""
        if not cls.AUTO_CLEANUP_ENABLED:
            return
        
        from datetime import datetime, timedelta
        
        cutoff_date = datetime.now() - timedelta(days=cls.KEEP_DAYS)
        cutoff_timestamp = int(cutoff_date.timestamp())
        
        cleaned_count = 0
        
        # Her rapor türü için temizlik yap
        for report_dir in [cls.INTEGRATED_REPORTS_DIR, cls.SAM2_REPORTS_DIR, 
                          cls.LLM_REPORTS_DIR, cls.DEBUG_REPORTS_DIR]:
            if not report_dir.exists():
                continue
                
            for file_path in report_dir.glob("*.json"):
                try:
                    # Dosya adından timestamp çıkar
                    parts = file_path.stem.split('_')
                    if len(parts) > 0 and parts[-1].isdigit():
                        file_timestamp = int(parts[-1])
                        if file_timestamp < cutoff_timestamp:
                            file_path.unlink()
                            cleaned_count += 1
                except (ValueError, IndexError):
                    continue
        
        if cleaned_count > 0:
            print(f"🧹 {cleaned_count} eski rapor temizlendi")
    
    @classmethod
    def list_reports(cls, report_type: str = "all") -> Dict[str, list]:
        """Mevcut raporları listele"""
        reports = {}
        
        type_mapping = {
            "integrated": cls.INTEGRATED_REPORTS_DIR,
            "sam2": cls.SAM2_REPORTS_DIR,
            "llm": cls.LLM_REPORTS_DIR,
            "debug": cls.DEBUG_REPORTS_DIR
        }
        
        if report_type == "all":
            target_dirs = type_mapping
        else:
            target_dirs = {report_type: type_mapping.get(report_type)}
        
        for type_name, report_dir in target_dirs.items():
            if type_name and report_dir and report_dir.exists():
                reports[type_name] = sorted([
                    str(f) for f in report_dir.glob("*.json")
                ], reverse=True)  # En yeni önce
            else:
                reports[type_name] = []
        
        return reports
    
    @classmethod
    def get_latest_report(cls, report_type: str) -> Optional[str]:
        """En son raporu al"""
        reports = cls.list_reports(report_type)
        type_reports = reports.get(report_type, [])
        return type_reports[0] if type_reports else None
    
    @classmethod
    def migrate_existing_reports(cls):
        """Mevcut raporları yeni dizin yapısına taşı"""
        moved_count = 0
        
        # Ana dizindeki entegre raporları taşı
        base_dir = cls.BASE_DIR
        pattern = "integrated_analysis_report_*.json"
        
        for old_report in base_dir.glob(pattern):
            new_path = cls.INTEGRATED_REPORTS_DIR / old_report.name
            if not new_path.exists():
                old_report.rename(new_path)
                moved_count += 1
                print(f"📦 Taşındı: {old_report.name} -> {new_path}")
        
        # Diğer rapor türleri için de benzer işlem yapılabilir
        if moved_count > 0:
            print(f"✅ {moved_count} rapor yeni dizin yapısına taşındı")
        else:
            print("ℹ️ Taşınacak rapor bulunamadı")

# Otomatik initialization
def initialize_analysis_config():
    """Analiz konfigürasyonunu başlat"""
    AnalysisConfig.initialize_directories()
    AnalysisConfig.migrate_existing_reports()
    AnalysisConfig.cleanup_old_reports()

if __name__ == "__main__":
    # Test ve örnek kullanım
    print("🔧 Analysis Config Test")
    print("=" * 40)
    
    initialize_analysis_config()
    
    # Örnek yol oluşturma
    print("\n📋 Örnek Yollar:")
    print(f"Entegre rapor: {AnalysisConfig.get_integrated_report_path()}")
    print(f"SAM2 rapor: {AnalysisConfig.get_sam2_report_path()}")
    print(f"LLM rapor: {AnalysisConfig.get_llm_report_path()}")
    print(f"Debug rapor: {AnalysisConfig.get_debug_report_path()}")
    print(f"Anomali görseli: {AnalysisConfig.get_anomaly_visualization_path('sn', 'wood')}")
    print(f"SAM2 çıktı: {AnalysisConfig.get_sam2_output_dir()}")
    
    # Mevcut raporları listele
    print("\n📊 Mevcut Raporlar:")
    reports = AnalysisConfig.list_reports()
    for report_type, file_list in reports.items():
        print(f"  {report_type}: {len(file_list)} adet")
        if file_list:
            latest = Path(file_list[0]).name
            print(f"    En son: {latest}")
