#!/usr/bin/env python3
"""
🎯 TAM ENTEGRASİON PİPELİNE'I:
Anomali Tespit → SAM2 Segmentasyon → LLM Analizi → Kullanıcı Onayı
"""

import torch
import time
import yaml
import numpy as np
import json
import cv2
import os
import sys
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple

# Kendi modüllerimizi import et
from models.SingleNet.trainer_sn import SnTrainer
from models.DBFAD.trainer_dbfad import DbfadTrainer
from models.EfficientAD.trainer_ead import EadTrainer
from models.ReverseDistillation.trainer_rd import RdTrainer
from models.StudentTeacher.trainer_st import StTrainer
from sam2_google_ai_pipeline import SAM2GoogleAIPipeline


class IntegratedAnomalySystem:
    """Entegre Anomali Tespit Sistemi: Anomali Tespit + SAM2 + LLM Analizi"""
    
    def __init__(self):
        """Sistemi başlat"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam2_pipeline = SAM2GoogleAIPipeline()
        self.supported_models = ["sn", "dbfad", "ead", "rd", "st"]
        self.threshold = 0.291  # Varsayılan threshold
        
        # 🧠 Adaptif threshold parametreleri (normalized anomaly score için)
        self.adaptive_threshold_config = {
            "use_adaptive": True,
            "methods": ["percentile", "statistical", "iqr"],
            "percentile_threshold": 95,  # 95th percentile
            "std_multiplier": 2.5,      # mean + 2.5*std
            "iqr_multiplier": 1.5,      # Q3 + 1.5*IQR
            "min_anomaly_ratio": 0.001, # Minimum %0.1 anomali oranı
            "combine_method": "mean"     # Methods: "mean", "max", "min"
        }
        
        print(f"🔧 Sistem başlatıldı - Cihaz: {self.device}")
        if self.device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"🧠 Adaptif threshold sistemi: {'✅ Aktif' if self.adaptive_threshold_config['use_adaptive'] else '❌ Pasif'}")
    
    def load_anomaly_model(self, model_type: str, config: Dict) -> object:
        """Anomali tespit modelini yükle"""
        print(f"📋 {model_type.upper()} modeli yükleniyor...")
        
        if model_type == "sn":
            return SnTrainer(config, self.device)
        elif model_type == "dbfad":
            return DbfadTrainer(config, self.device)
        elif model_type == "ead":
            return EadTrainer(config, self.device)
        elif model_type == "rd":
            return RdTrainer(config, self.device)
        elif model_type == "st":
            return StTrainer(config, self.device)
        else:
            raise ValueError(f"Desteklenmeyen model türü: {model_type}")
    
    def preprocess_image(self, image_path: str, img_size: int = 224) -> torch.Tensor:
        """Görüntüyü inference için hazırla"""
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def get_available_models_by_dataset(self) -> Dict[str, List[str]]:
        """Dataset'lere göre mevcut modelleri listele"""
        results_dir = Path("results/models")
        if not results_dir.exists():
            return {}

        available_models = {}
        for dataset_dir in results_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                available_models[dataset_name] = []

                for model_dir in dataset_dir.glob("*/*.pth"):
                    model_type = model_dir.parent.name
                    if model_type not in available_models[dataset_name]:
                        available_models[dataset_name].append(model_type)

        return available_models
    
    def detect_anomalies(self, image_path: str, model_type: str, dataset: str) -> Dict:
        """1️⃣ ADIM: Anomali tespiti yap - Normalized Anomaly Score ile"""
        print("\n" + "="*60)
        print("🔍 ADIM 1: ANOMALİ TESPİTİ (Normalized Anomaly Score)")
        print("="*60)
        
        try:
            # Import the anomaly calculation function
            from utils.functions import cal_anomaly_maps
            
            # Konfigürasyon hazırla - model_dir yolu düzeltildi
            config = {
                "data_path": f"./dataset",  # Base dataset path only
                "obj": dataset,  # This will be added by trainer
                "save_path": "./results",
                "distillType": model_type,
                "inference_only_mode": True,  # Sadece inference için
                "TrainingData": {
                    "epochs": 100,
                    "batch_size": 32,
                    "lr": 0.0004,
                    "img_size": 224,
                    "crop_size": 224,
                    "norm": True
                }
            }
            
            # Model yükle
            trainer = self.load_anomaly_model(model_type, config)
            
            # Model directory'yi manuel olarak düzelt
            trainer.model_dir = f"./results/models/{dataset}/{model_type}"
            
            model_path = Path(f"results/models/{dataset}/{model_type}/student.pth")
            
            if not model_path.exists():
                return {
                    "success": False, 
                    "error": f"Model ağırlıkları bulunamadı: {model_path}"
                }
            
            trainer.load_weights()
            trainer.change_mode("eval")  # Set to evaluation mode
            print(f"✅ Model yüklendi: {model_path}")
            
            # Görüntüyü işle
            image_tensor = self.preprocess_image(image_path)
            print(f"✅ Görüntü işlendi: {image_path}")
            
            # Inference yap
            with torch.no_grad():
                trainer.infer(image_tensor)
                trainer.post_process()  # This processes the features
                
                # Now calculate the actual anomaly map using the processed features
                # Make sure we pass the features correctly (not as lists)
                anomaly_map = cal_anomaly_maps(
                    trainer.features_s,  # Remove the list wrapper
                    trainer.features_t,  # Remove the list wrapper
                    out_size=224,  # Same as crop_size
                    norm=trainer.norm
                )
            
            # Anomaly map'i numpy array'e çevir (eğer tek boyutluysa)
            if len(anomaly_map.shape) == 3:
                anomaly_map = anomaly_map[0]  # Remove batch dimension
            
            # 📊 Temel anomali skorlarını hesapla
            max_anomaly = np.max(anomaly_map)
            mean_anomaly = np.mean(anomaly_map)
            std_anomaly = np.std(anomaly_map)
            
            print(f"📊 Temel Anomali İstatistikleri:")
            print(f"   📏 Min/Max: [{np.min(anomaly_map):.6f}, {max_anomaly:.6f}]")
            print(f"   📈 Mean ± Std: {mean_anomaly:.6f} ± {std_anomaly:.6f}")
            print(f"   📊 Orijinal Threshold: {self.threshold:.6f}")
            
            # 🧠 Adaptif threshold hesapla (eğer aktifse)
            adaptive_results = {}
            final_threshold = self.threshold
            threshold_method = "Orijinal (Sabit)"
            
            if self.adaptive_threshold_config["use_adaptive"]:
                adaptive_results = self.calculate_adaptive_threshold(anomaly_map)
                
                # Adaptif threshold daha iyi mi kontrol et
                if adaptive_results["is_adaptive_better"]:
                    final_threshold = adaptive_results["adaptive_threshold"]
                    threshold_method = "Adaptif (Normalized)"
                    print(f"🧠 Adaptif threshold kullanılacak: {final_threshold:.6f}")
                else:
                    print(f"⚠️  Adaptif threshold kaliteli değil, orijinal threshold kullanılıyor: {self.threshold:.6f}")
            
            # Final anomali hesaplamaları
            anomaly_pixels = np.sum(anomaly_map > final_threshold)
            total_pixels = anomaly_map.size
            anomaly_ratio = anomaly_pixels / total_pixels
            
            print(f"\n🎯 Final Anomali Analizi ({threshold_method}):")
            print(f"   🔴 Threshold: {final_threshold:.6f}")
            print(f"   📊 Anomali pikselleri: {anomaly_pixels:,}/{total_pixels:,} ({anomaly_ratio:.4%})")
            
            # 🔍 Normalized Anomaly Score hesapla
            normalized_scores = None
            if adaptive_results and "normalized_scores" in adaptive_results:
                normalized_scores = adaptive_results["normalized_scores"]
                normalized_threshold = adaptive_results["normalized_threshold"]
                
                # Normalized skorlarda anomali oranı
                normalized_anomaly_count = np.sum(normalized_scores > normalized_threshold)
                normalized_anomaly_ratio = normalized_anomaly_count / len(normalized_scores)
                
                print(f"🧠 Normalized Anomaly Score Analizi:")
                print(f"   📈 Z-score threshold: {normalized_threshold:.3f}")
                print(f"   🔴 Normalized anomali oranı: {normalized_anomaly_ratio:.4%}")
            
            # Anomali var mı? (Daha hassas kriter)
            has_anomaly = (
                max_anomaly > final_threshold and 
                anomaly_ratio > self.adaptive_threshold_config["min_anomaly_ratio"]
            )
            
            # Anomali seviyesi belirleme
            if anomaly_ratio < 0.001:
                severity = "NORMAL"
                severity_emoji = "✅"
            elif anomaly_ratio < 0.01:
                severity = "DÜŞÜK RİSK"
                severity_emoji = "⚠️"
            elif anomaly_ratio < 0.05:
                severity = "ORTA RİSK"
                severity_emoji = "🟠"
            else:
                severity = "YÜKSEK RİSK"
                severity_emoji = "🚨"
            
            result = {
                "success": True,
                "has_anomaly": has_anomaly,
                "max_score": float(max_anomaly),
                "mean_score": float(mean_anomaly),
                "std_score": float(std_anomaly),
                "threshold": final_threshold,
                "threshold_method": threshold_method,
                "anomaly_ratio": float(anomaly_ratio),
                "anomaly_map": anomaly_map,
                "model_type": model_type,
                "dataset": dataset,
                "severity": severity,
                "severity_emoji": severity_emoji,
                "adaptive_results": adaptive_results if self.adaptive_threshold_config["use_adaptive"] else None,
                "normalized_scores": normalized_scores.tolist() if normalized_scores is not None else None
            }
            
            if has_anomaly:
                print(f"🔴 ANOMALİ TESPİT EDİLDİ! {severity_emoji} {severity}")
            else:
                print(f"🟢 Anomali tespit edilmedi - {severity}")
            
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()  # Print full error trace for debugging
            return {"success": False, "error": f"Anomali tespit hatası: {str(e)}"}
    
    def calculate_adaptive_threshold(self, anomaly_map: np.ndarray) -> Dict:
        """🧠 Adaptif threshold hesapla - normalized anomaly score mantığı"""
        print("\n🧮 Adaptif Threshold Hesaplaması (Normalized Anomaly Score)")
        print("-" * 60)
        
        # Temel istatistikler
        flat_scores = anomaly_map.flatten()
        mean_score = np.mean(flat_scores)
        std_score = np.std(flat_scores)
        median_score = np.median(flat_scores)
        
        # Quartile'lar
        q1 = np.percentile(flat_scores, 25)
        q3 = np.percentile(flat_scores, 75)
        iqr = q3 - q1
        
        print(f"📊 Temel İstatistikler:")
        print(f"   📏 Min/Max: [{np.min(flat_scores):.6f}, {np.max(flat_scores):.6f}]")
        print(f"   📈 Mean ± Std: {mean_score:.6f} ± {std_score:.6f}")
        print(f"   📊 Median: {median_score:.6f}")
        print(f"   📦 Q1/Q3: [{q1:.6f}, {q3:.6f}], IQR: {iqr:.6f}")
        
        thresholds = {}
        
        # 1. Percentile bazlı threshold
        percentile_thresh = np.percentile(flat_scores, self.adaptive_threshold_config["percentile_threshold"])
        thresholds["percentile"] = percentile_thresh
        
        # 2. İstatistiksel threshold (mean + k*std)
        statistical_thresh = mean_score + (self.adaptive_threshold_config["std_multiplier"] * std_score)
        thresholds["statistical"] = statistical_thresh
        
        # 3. IQR bazlı threshold
        iqr_thresh = q3 + (self.adaptive_threshold_config["iqr_multiplier"] * iqr)
        thresholds["iqr"] = iqr_thresh
        
        # Threshold'ları birleştir
        selected_methods = self.adaptive_threshold_config["methods"]
        selected_thresholds = [thresholds[method] for method in selected_methods if method in thresholds]
        
        if self.adaptive_threshold_config["combine_method"] == "mean":
            final_threshold = np.mean(selected_thresholds)
        elif self.adaptive_threshold_config["combine_method"] == "max":
            final_threshold = np.max(selected_thresholds)
        elif self.adaptive_threshold_config["combine_method"] == "min":
            final_threshold = np.min(selected_thresholds)
        else:
            final_threshold = np.mean(selected_thresholds)
        
        # Minimum threshold kontrolü (çok düşük olmasın)
        min_reasonable_threshold = mean_score + 0.5 * std_score
        final_threshold = max(final_threshold, min_reasonable_threshold)
        
        # 🎯 Normalized Anomaly Score hesaplama
        normalized_scores = (flat_scores - mean_score) / (std_score + 1e-8)  # Z-score normalizasyonu
        normalized_threshold = (final_threshold - mean_score) / (std_score + 1e-8)
        
        # Sonuçları hesapla
        adaptive_anomaly_pixels = np.sum(flat_scores > final_threshold)
        adaptive_anomaly_ratio = adaptive_anomaly_pixels / len(flat_scores)
        
        # Orijinal threshold ile karşılaştırma
        original_anomaly_pixels = np.sum(flat_scores > self.threshold)
        original_anomaly_ratio = original_anomaly_pixels / len(flat_scores)
        
        print(f"\n🔬 Threshold Hesaplamaları:")
        print(f"   📊 Percentile ({self.adaptive_threshold_config['percentile_threshold']}%): {percentile_thresh:.6f}")
        print(f"   📈 İstatistiksel (μ + {self.adaptive_threshold_config['std_multiplier']}σ): {statistical_thresh:.6f}")
        print(f"   📦 IQR (Q3 + {self.adaptive_threshold_config['iqr_multiplier']}*IQR): {iqr_thresh:.6f}")
        print(f"   🎯 Final Adaptif Threshold: {final_threshold:.6f}")
        print(f"   📏 Orijinal Threshold: {self.threshold:.6f}")
        print(f"   🧠 Normalized Threshold (Z-score): {normalized_threshold:.3f}")
        
        print(f"\n📈 Karşılaştırma:")
        print(f"   🔴 Adaptif → Anomali oranı: {adaptive_anomaly_ratio:.4%} ({adaptive_anomaly_pixels:,} piksel)")
        print(f"   🔵 Orijinal → Anomali oranı: {original_anomaly_ratio:.4%} ({original_anomaly_pixels:,} piksel)")
        
        # Hangi yöntemin daha mantıklı olduğunu değerlendir
        is_adaptive_better = self.evaluate_threshold_quality(flat_scores, final_threshold, self.threshold)
        
        return {
            "adaptive_threshold": final_threshold,
            "original_threshold": self.threshold,
            "normalized_threshold": normalized_threshold,
            "normalized_scores": normalized_scores,
            "method_thresholds": thresholds,
            "adaptive_anomaly_ratio": adaptive_anomaly_ratio,
            "original_anomaly_ratio": original_anomaly_ratio,
            "is_adaptive_better": is_adaptive_better,
            "statistics": {
                "mean": mean_score,
                "std": std_score,
                "median": median_score,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "min": np.min(flat_scores),
                "max": np.max(flat_scores)
            }
        }
    
    def evaluate_threshold_quality(self, scores: np.ndarray, adaptive_thresh: float, original_thresh: float) -> bool:
        """🔍 Threshold kalitesini değerlendir - normalized anomaly score kriterlerine göre"""
        
        # Adaptif threshold ile tespit edilen anomaliler
        adaptive_anomalies = scores > adaptive_thresh
        adaptive_ratio = np.mean(adaptive_anomalies)
        
        # Orijinal threshold ile tespit edilen anomaliler
        original_anomalies = scores > original_thresh
        original_ratio = np.mean(original_anomalies)
        
        # Normalized score kalite kriterleri
        criteria = {
            "reasonable_ratio": 0.001 <= adaptive_ratio <= 0.15,  # %0.1 - %15 arası makul
            "not_too_sensitive": adaptive_ratio < 0.5,  # Çok hassas olmasın
            "captures_outliers": adaptive_thresh > np.percentile(scores, 90),  # En üst %10'u yakalasın
            "better_than_original": adaptive_ratio > 0 and abs(adaptive_ratio - original_ratio) > 0.001,
            "statistical_significance": adaptive_thresh > np.mean(scores) + 2 * np.std(scores)  # İstatistiksel anlamlılık
        }
        
        quality_score = sum(criteria.values())
        is_better = quality_score >= 3  # 5'ten en az 3'ü sağlanmalı
        
        print(f"\n🔍 Threshold Kalite Değerlendirmesi:")
        for criterion, passed in criteria.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {criterion.replace('_', ' ').title()}")
        print(f"   🎯 Kalite Skoru: {quality_score}/5 - {'🟢 Adaptif Daha İyi' if is_better else '🔵 Orijinal Kullan'}")
        
        return is_better
    
    def calculate_normalized_anomaly_scores(self, anomaly_map: np.ndarray, threshold: float) -> np.ndarray:
        """Normalized anomaly scores hesapla (Z-score)"""
        mean = np.mean(anomaly_map)
        std = np.std(anomaly_map)
        
        # Z-score hesapla
        z_scores = (anomaly_map - mean) / std
        
        # Anomali skoru eşik değerine göre ayarla
        adjusted_scores = np.where(z_scores > threshold, z_scores, 0)
        
        return adjusted_scores
    
    def create_anomaly_points_for_sam2(self, anomaly_map: np.ndarray, image_path: str, 
                                      top_k: int = 5) -> Dict:
        """Anomali haritasından SAM2 için nokta koordinatları oluştur"""
        print("\n📍 SAM2 için anomali noktaları oluşturuluyor...")
        
        # En yüksek anomali skorlarına sahip noktaları bul
        flat_indices = np.argpartition(anomaly_map.ravel(), -top_k)[-top_k:]
        indices_2d = np.unravel_index(flat_indices, anomaly_map.shape)
        
        # Orijinal görüntü boyutlarını al
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError("Orijinal görüntü yüklenemedi")
        
        orig_h, orig_w = original_img.shape[:2]
        map_h, map_w = anomaly_map.shape
        
        # Koordinatları orijinal görüntü boyutlarına ölçekle
        points = []
        for i in range(len(flat_indices)):
            y, x = indices_2d[0][i], indices_2d[1][i]
            
            # Ölçekleme
            orig_x = int(x * orig_w / map_w)
            orig_y = int(y * orig_h / map_h)
            
            # Anomali skorunu al
            score = anomaly_map[y, x]
            
            points.append({
                "x": orig_x,
                "y": orig_y,
                "score": float(score),
                "label": True  # Pozitif nokta
            })
        
        # En yüksek skordan düşük skora sırala
        points.sort(key=lambda p: p["score"], reverse=True)
        
        print(f"✅ {len(points)} anomali noktası oluşturuldu")
        for i, point in enumerate(points):
            print(f"   {i+1}. Nokta: ({point['x']}, {point['y']}) - Skor: {point['score']:.4f}")
        
        # SAM2 formatında prompt oluştur
        sam2_prompts = {
            "prompts": [
                {
                    "type": "points",
                    "object_id": 1,
                    "frame_index": 0,
                    "points": points[:3]  # İlk 3 noktayı kullan
                }
            ]
        }
        
        return sam2_prompts
    
    def visualize_anomaly_detection(self, original_image_path: str, anomaly_map: np.ndarray, 
                                   threshold: float, save_path: str = None) -> str:
        """Anomali tespit sonuçlarını görselleştir"""
        plt.figure(figsize=(15, 5))
        
        # Orijinal görüntü
        original_image = cv2.imread(original_image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        plt.subplot(131)
        plt.imshow(original_image)
        plt.title('Orijinal Görüntü')
        plt.axis('off')
        
        # Anomali haritası
        plt.subplot(132)
        plt.imshow(anomaly_map, cmap='jet')
        plt.colorbar()
        plt.title('Anomali Haritası')
        plt.axis('off')
        
        # Eşiklenmiş sonuç
        plt.subplot(133)
        thresholded = (anomaly_map > threshold).astype(np.float32)
        plt.imshow(thresholded, cmap='gray')
        plt.title(f'Eşiklenmiş (>{threshold:.3f})')
        plt.axis('off')
        
        plt.tight_layout()
        
        if not save_path:
            save_path = f"results/anomaly_detection_{int(time.time())}.png"
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        print(f"💾 Anomali görselleştirmesi kaydedildi: {save_path}")
        return save_path
    
    def run_sam2_segmentation(self, image_path: str, anomaly_result: Dict) -> Dict:
        """2️⃣ ADIM: SAM2 ile segmentasyon"""
        print("\n" + "="*60)
        print("🎯 ADIM 2: SAM2 SEGMENTASYON")
        print("="*60)
        
        try:
            # Anomali noktalarını SAM2 formatına çevir
            sam2_prompts = self.create_anomaly_points_for_sam2(
                anomaly_result["anomaly_map"], 
                image_path
            )
            
            # SAM2 pipeline'ını çalıştır
            sam2_result = self.sam2_pipeline.run_sam2_segmentation(
                image_path=image_path,
                prompts=sam2_prompts,
                output_dir=f"temp_analysis/sam2_output_{int(time.time())}"
            )
            
            if sam2_result["success"]:
                print("✅ SAM2 segmentasyonu başarılı!")
                
                # Çıktı dosyalarını bul
                output_images = self.sam2_pipeline.find_output_images(sam2_result["output_dir"])
                sam2_result["output_images"] = output_images
                
                print(f"📁 Çıktı dosyaları:")
                for img_type, img_path in output_images.items():
                    print(f"   {img_type}: {img_path}")
            else:
                print(f"❌ SAM2 segmentasyonu başarısız: {sam2_result.get('error')}")
            
            return sam2_result
            
        except Exception as e:
            return {"success": False, "error": f"SAM2 segmentasyon hatası: {str(e)}"}
    
    def run_llm_analysis(self, original_image: str, sam2_result: Dict) -> Dict:
        """3️⃣ ADIM: LLM ile analiz"""
        print("\n" + "="*60)
        print("🤖 ADIM 3: LLM ANALİZİ")
        print("="*60)
        
        try:
            # Segmente edilmiş görüntüyü bul
            output_images = sam2_result.get("output_images", {})
            segmented_image = output_images.get('segmented') or output_images.get('result')
            
            if not segmented_image:
                print("⚠️ Segmente edilmiş görüntü bulunamadı, sadece orijinal görüntü analiz edilecek")
            
            # LLM analizi yap
            llm_result = self.sam2_pipeline.analyze_with_google_ai(
                image_path=original_image,
                analysis_type="wood_inspection",
                segmented_image=segmented_image
            )
            
            if llm_result.get("success"):
                print("✅ LLM analizi başarılı!")
                
                # Analiz sonuçlarını göster
                analysis = llm_result.get("analysis", {})
                if isinstance(analysis, dict):
                    print(f"📊 ANALIZ SONUÇLARI:")
                    
                    # Ana bulgular
                    if "anomali_turu" in analysis:
                        print(f"   🔍 Anomali Türü: {analysis['anomali_turu']}")
                    if "siddet_seviyesi" in analysis:
                        print(f"   ⚡ Ciddiyet: {analysis['siddet_seviyesi']}")
                    if "guven_skoru" in analysis:
                        print(f"   📈 Güven Skoru: {analysis['guven_skoru']}%")
                    if "recommended_action" in analysis:
                        print(f"   💡 Önerilen Aksiyon: {analysis['recommended_action']}")
                    if "estimated_cost" in analysis:
                        print(f"   💰 Maliyet Etkisi: {analysis['estimated_cost']}")
            else:
                print(f"❌ LLM analizi başarısız: {llm_result.get('error')}")
            
            return llm_result
            
        except Exception as e:
            return {"success": False, "error": f"LLM analizi hatası: {str(e)}"}
    
    def ask_user_confirmation(self, step: str, details: str = "") -> bool:
        """4️⃣ ADIM: Kullanıcı onayı iste"""
        print(f"\n⏸️  KULLANICI ONAYI - {step}")
        print("-" * 40)
        if details:
            print(details)
        
        while True:
            response = input("\nDevam etmek istiyor musunuz? (e/h/çıkış): ").lower().strip()
            if response in ['e', 'evet', 'y', 'yes']:
                return True
            elif response in ['h', 'hayır', 'n', 'no']:
                return False
            elif response in ['çıkış', 'exit', 'quit', 'q']:
                print("🚪 İşlem kullanıcı tarafından sonlandırıldı.")
                sys.exit(0)
            else:
                print("Lütfen 'e' (evet), 'h' (hayır) veya 'çıkış' yazın.")
    
    def run_full_pipeline(self, image_path: str, model_type: str = "sn", 
                         dataset: str = "wood") -> Dict:
        """🎯 TAM PİPELİNE'I ÇALIŞTIR"""
        print("\n" + "🎯" + "="*58 + "🎯")
        print("        TAM ENTEGRASİON PİPELİNE'I BAŞLADI")
        print("🎯" + "="*58 + "🎯")
        print(f"📁 Görüntü: {image_path}")
        print(f"🤖 Model: {model_type.upper()}")
        print(f"📊 Dataset: {dataset}")
        
        # Dosya kontrolü
        if not Path(image_path).exists():
            return {"success": False, "error": f"Görüntü dosyası bulunamadı: {image_path}"}
        
        pipeline_results = {
            "timestamp": int(time.time()),
            "input_image": image_path,
            "model_type": model_type,
            "dataset": dataset,
            "steps": {},
            "success": False
        }
        
        try:
            # 1️⃣ ANOMALİ TESPİTİ
            anomaly_result = self.detect_anomalies(image_path, model_type, dataset)
            pipeline_results["steps"]["anomaly_detection"] = anomaly_result
            
            if not anomaly_result["success"]:
                return pipeline_results
            
            # Anomali görselleştir
            if anomaly_result["has_anomaly"]:
                viz_path = self.visualize_anomaly_detection(
                    image_path, 
                    anomaly_result["anomaly_map"], 
                    anomaly_result["threshold"]
                )
                pipeline_results["anomaly_visualization"] = viz_path
            
            # Kullanıcı onayı - Anomali tespit
            details = f"Anomali tespit edildi: {'EVET' if anomaly_result['has_anomaly'] else 'HAYIR'}\n"
            details += f"Maksimum skor: {anomaly_result['max_score']:.4f}\n"
            details += f"Anomali oranı: {anomaly_result['anomaly_ratio']:.2%}"
            
            if not self.ask_user_confirmation("ANOMALI TESPİTİ TAMAMLANDI", details):
                print("🛑 İşlem kullanıcı tarafından durduruldu (Anomali tespit sonrası)")
                return pipeline_results
            
            # Eğer anomali yoksa pipeline'ı sonlandır
            if not anomaly_result["has_anomaly"]:
                print("ℹ️ Anomali tespit edilmediği için segmentasyon yapılmayacak")
                pipeline_results["success"] = True
                pipeline_results["message"] = "Anomali tespit edilmedi"
                return pipeline_results
            
            # 2️⃣ SAM2 SEGMENTASYON
            sam2_result = self.run_sam2_segmentation(image_path, anomaly_result)
            pipeline_results["steps"]["sam2_segmentation"] = sam2_result
            
            if not sam2_result["success"]:
                print(f"⚠️ SAM2 başarısız, ancak devam ediliyor: {sam2_result.get('error')}")
            
            # Kullanıcı onayı - SAM2 segmentasyon
            details = f"SAM2 segmentasyon: {'BAŞARILI' if sam2_result['success'] else 'BAŞARISIZ'}\n"
            if sam2_result["success"] and "output_images" in sam2_result:
                details += f"Çıktı dosya sayısı: {len(sam2_result['output_images'])}"
            
            if not self.ask_user_confirmation("SAM2 SEGMENTASYON TAMAMLANDI", details):
                print("🛑 İşlem kullanıcı tarafından durduruldu (SAM2 sonrası)")
                return pipeline_results
            
            # 3️⃣ LLM ANALİZİ
            llm_result = self.run_llm_analysis(image_path, sam2_result)
            pipeline_results["steps"]["llm_analysis"] = llm_result
            
            # Kullanıcı onayı - LLM analizi
            details = f"LLM analizi: {'BAŞARILI' if llm_result.get('success') else 'BAŞARISIZ'}\n"
            if llm_result.get("success"):
                analysis = llm_result.get("analysis", {})
                if "recommended_action" in analysis:
                    details += f"Önerilen aksiyon: {analysis['recommended_action']}"
            
            if not self.ask_user_confirmation("LLM ANALİZİ TAMAMLANDI", details):
                print("🛑 İşlem kullanıcı tarafından durduruldu (LLM sonrası)")
                return pipeline_results
            
            # 4️⃣ RAPOR OLUŞTURMA
            report_path = f"integrated_analysis_report_{pipeline_results['timestamp']}.json"
            self.create_integrated_report(pipeline_results, report_path)
            pipeline_results["report_path"] = report_path
            
            # Başarı durumu
            pipeline_results["success"] = True
            
            print("\n" + "🎉" + "="*58 + "🎉")
            print("        TAM PİPELİNE BAŞARIYLA TAMAMLANDI!")
            print("🎉" + "="*58 + "🎉")
            
            return pipeline_results
            
        except KeyboardInterrupt:
            print("\n⚠️ İşlem kullanıcı tarafından iptal edildi.")
            pipeline_results["error"] = "Kullanıcı iptali"
            return pipeline_results
        except Exception as e:
            print(f"\n❌ Pipeline hatası: {str(e)}")
            pipeline_results["error"] = str(e)
            return pipeline_results
    
    def create_integrated_report(self, results: Dict, output_path: str):
        """Entegre rapor oluştur"""
        print(f"\n📄 Entegre rapor oluşturuluyor: {output_path}")
        
        # Rapor hazırla - JSON serializable hale getir
        def make_serializable(obj):
            """Objeyi JSON serializable hale getir - geliştirilmiş versiyon"""
            if obj is None:
                return None
            elif isinstance(obj, (str, int, float)):
                return obj
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool, np.bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # Object'i dict'e çevir
                return make_serializable(obj.__dict__)
            else:
                # Son çare: string'e çevir
                try:
                    return str(obj)
                except Exception as e:
                    print(f"⚠️ Serialization warning for {type(obj)}: {e}")
                    return f"<{type(obj).__name__}>"
        
        try:
            # Tüm step_results'ı serializable hale getir
            print("🔧 JSON serialization işlemi başlatılıyor...")
            serializable_steps = make_serializable(results.get("steps", {}))
            
            report = {
                "pipeline_info": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(results["timestamp"])),
                    "input_image": str(results["input_image"]),
                    "model_type": str(results["model_type"]),
                    "dataset": str(results["dataset"]),
                    "success": bool(results["success"])
                },
                "step_results": serializable_steps,
                "error": str(results.get("error")) if results.get("error") else None,
                "report_path": str(output_path)
            }
            
            # Raporu kaydet
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Rapor kaydedildi: {output_path}")
            
            # Özet göster
            print(f"\n📋 PİPELİNE ÖZETİ:")
            print(f"   🔍 Anomali tespit: {'✅' if summary['anomaly_detected'] else '❌'}")
            print(f"   🎯 Segmentasyon: {'✅' if summary['segmentation_performed'] else '❌'}")
            print(f"   🤖 LLM analizi: {'✅' if summary['llm_analysis_completed'] else '❌'}")
            print(f"   💡 Final öneri: {summary['final_recommendation']}")
        
        except Exception as e:
            print(f"❌ Rapor oluşturma hatası: {e}")


def main():
    """Ana fonksiyon - Kullanıcı arayüzü"""
    print("🎯 ENTEGRE ANOMALİ TESPİT SİSTEMİ")
    print("=" * 50)
    print("Anomali Tespit → SAM2 Segmentasyon → LLM Analizi")
    print("=" * 50)
    
    # Sistem başlat
    system = IntegratedAnomalySystem()
    
    # Mevcut modelleri listele
    available_models = system.get_available_models_by_dataset()
    if not available_models:
        print("❌ Hiç eğitilmiş model bulunamadı!")
        print("Önce train.py ile model eğitmeniz gerekiyor.")
        return
    
    print("\n📊 MEVCUT MODELLERİ:")
    for dataset, models in available_models.items():
        print(f"   {dataset}: {', '.join(models)}")
    
    # Parametreleri al
    if len(sys.argv) >= 2:
        # Komut satırından
        image_path = sys.argv[1]
        model_type = sys.argv[2] if len(sys.argv) > 2 else "sn"
        dataset = sys.argv[3] if len(sys.argv) > 3 else "wood"
    else:
        # İnteraktif mod
        print("\n📝 PARAMETRELER:")
        
        # Görüntü yolu
        image_path = input("Görüntü yolunu girin (örn: dataset/wood/test/hole/002.png): ").strip()
        if not image_path:
            image_path = "dataset/wood/test/hole/002.png"  # Varsayılan
        
        # Dataset seç
        datasets = list(available_models.keys())
        print(f"\nMevcut dataset'ler: {', '.join(datasets)}")
        dataset = input(f"Dataset seçin (varsayılan: wood): ").strip()
        if not dataset:
            dataset = "wood"
        
        # Model seç
        if dataset in available_models:
            models = available_models[dataset]
            print(f"\n{dataset} için mevcut modeller: {', '.join(models)}")
            model_type = input(f"Model seçin (varsayılan: {models[0] if models else 'sn'}): ").strip()
            if not model_type:
                model_type = models[0] if models else "sn"
        else:
            print(f"⚠️ {dataset} dataset'i için model bulunamadı, varsayılan 'sn' kullanılacak")
            model_type = "sn"
    
    # Parametreleri doğrula
    print(f"\n🔧 SEÇILEN PARAMETRELER:")
    print(f"   📁 Görüntü: {image_path}")
    print(f"   🤖 Model: {model_type}")
    print(f"   📊 Dataset: {dataset}")
    
    # Dosya varlığını kontrol et
    if not Path(image_path).exists():
        print(f"\n❌ HATA: Görüntü dosyası bulunamadı: {image_path}")
        return
    
    # Final onay
    if not system.ask_user_confirmation("BAŞLANGICA HAZIR", "Pipeline'ı başlatmaya hazır mısınız?"):
        print("🚪 İşlem iptal edildi.")
        return
    
    # Pipeline'ı çalıştır
    try:
        results = system.run_full_pipeline(
            image_path=image_path,
            model_type=model_type,
            dataset=dataset
        )
        
        # Sonuç özeti
        print(f"\n📋 FINAL SONUÇLAR:")
        print(f"   ✅ Başarılı: {'EVET' if results['success'] else 'HAYIR'}")
        if "report_path" in results:
            print(f"   📄 Rapor: {results['report_path']}")
        if results.get("error"):
            print(f"   ❌ Hata: {results['error']}")
        
        # Temizlik seçeneği
        cleanup = input("\nGeçici dosyalar silinsin mi? (e/h, varsayılan: e): ").lower()
        if cleanup != 'h':
            system.sam2_pipeline.cleanup_temp_files()
            print("🧹 Geçici dosyalar temizlendi")
    
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
    
    print("\n🎯 Entegre sistem tamamlandı!")


if __name__ == "__main__":
    main()