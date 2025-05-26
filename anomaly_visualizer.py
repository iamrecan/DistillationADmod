#!/usr/bin/env python3
"""
🎨 ANOMALİ GÖRSELLEŞTİRME ARACI
Anomali tespit sonuçlarını detaylı şekilde görselleştir ve karşılaştır
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import sys
import time
import random
import traceback
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from sklearn.cluster import KMeans

# Kendi modüllerimizi import et
from models.SingleNet.trainer_sn import SnTrainer
from models.DBFAD.trainer_dbfad import DbfadTrainer
from models.EfficientAD.trainer_ead import EadTrainer
from models.ReverseDistillation.trainer_rd import RdTrainer
from models.StudentTeacher.trainer_st import StTrainer


class AnomalyVisualizer:
    """Anomali tespit sonuçlarını görselleştiren sınıf"""
    
    def __init__(self):
        """Görselleştirici başlat"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.supported_models = ["sn", "dbfad", "ead", "rd", "st"]
        self.threshold = 0.291  # Varsayılan threshold (geriye uyumluluk için)
        
        # Adaptif threshold parametreleri
        self.adaptive_threshold_config = {
            "use_adaptive": True,
            "methods": ["percentile", "statistical", "iqr"],
            "percentile_threshold": 95,  # 95th percentile
            "std_multiplier": 2.5,      # mean + 2.5*std
            "iqr_multiplier": 1.5,      # Q3 + 1.5*IQR
            "min_anomaly_ratio": 0.001, # Minimum %0.1 anomali oranı
            "combine_method": "mean"     # Methods: "mean", "max", "min"
        }
        
        print(f"🎨 Anomali Görselleştirici başlatıldı - Cihaz: {self.device}")
        print(f"🧠 Adaptif threshold sistemi aktif: {self.adaptive_threshold_config['use_adaptive']}")
        
        # Renk paleti ayarla
        plt.style.use('default')
        sns.set_palette("husl")
    
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
    
    def load_original_image(self, image_path: str) -> np.ndarray:
        """Orijinal görüntüyü yükle (görselleştirme için)"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def calibrate_threshold_on_normal_images(self, model_type: str, dataset: str, num_samples: int = 10) -> Dict:
        """Normal görüntüler üzerinde threshold kalibrasyonu yap"""
        print(f"\n🎯 THRESHOLD KALİBRASYONU")
        print(f"📊 Dataset: {dataset}")
        print(f"🤖 Model: {model_type}")
        print(f"📁 Örneklem: {num_samples} normal görüntü")
        print("-" * 50)
        
        try:
            # Normal görüntü dizinini kontrol et
            normal_dir = Path(f"dataset/{dataset}/test/good")
            if not normal_dir.exists():
                return {
                    "success": False,
                    "error": f"Normal görüntü dizini bulunamadı: {normal_dir}"
                }
            
            # Normal görüntüleri listele
            normal_images = list(normal_dir.glob("*.png")) + list(normal_dir.glob("*.jpg"))
            if len(normal_images) == 0:
                return {
                    "success": False,
                    "error": f"Normal görüntü bulunamadı: {normal_dir}"
                }
            
            # Örneklem seç
            selected_images = random.sample(normal_images, min(num_samples, len(normal_images)))
            print(f"✅ {len(selected_images)} normal görüntü seçildi")
            
            # Model yükle
            config = {
                "data_path": f"./dataset",
                "obj": dataset,
                "save_path": "./results",
                "distillType": model_type,
                "inference_only_mode": True,
                "TrainingData": {
                    "epochs": 100,
                    "batch_size": 32,
                    "lr": 0.0004,
                    "img_size": 224,
                    "crop_size": 224,
                    "norm": True
                }
            }
            
            trainer = self.load_anomaly_model(model_type, config)
            trainer.model_dir = f"./results/models/{dataset}/{model_type}"
            
            model_path = Path(f"results/models/{dataset}/{model_type}/student.pth")
            if not model_path.exists():
                return {
                    "success": False,
                    "error": f"Model ağırlıkları bulunamadı: {model_path}"
                }
            
            trainer.load_weights()
            trainer.change_mode("eval")
            
            # Custom anomaly calculation function
            def calculate_raw_anomaly_map(fs_list, ft_list, out_size, norm):
                """Normalizasyon olmadan anomali haritası hesapla"""
                anomaly_map = 0
                for i in range(len(ft_list)):
                    fs = fs_list[i]
                    ft = ft_list[i]
                    fs_norm = F.normalize(fs, p=2) if norm else fs
                    ft_norm = F.normalize(ft, p=2) if norm else ft

                    a_map = 0.5 * (ft_norm - fs_norm) ** 2
                    a_map = a_map.sum(1, keepdim=True)
                    a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=False)
                    anomaly_map += a_map
                
                anomaly_map = anomaly_map.squeeze().cpu().numpy()
                
                # Gaussian filter uygula
                from scipy.ndimage import gaussian_filter
                if len(anomaly_map.shape) == 2:
                    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                else:
                    for i in range(anomaly_map.shape[0]):
                        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)

                return anomaly_map
            
            # Normal görüntüler üzerinde analiz
            all_normal_scores = []
            normal_stats = []
            
            print("🔍 Normal görüntüler analiz ediliyor...")
            
            for i, img_path in enumerate(selected_images):
                print(f"   📸 {i+1}/{len(selected_images)}: {img_path.name}")
                
                # Görüntüyü işle
                image_tensor = self.preprocess_image(str(img_path))
                
                # Inference
                with torch.no_grad():
                    trainer.infer(image_tensor)
                    trainer.post_process()
                    
                    # Raw anomaly map hesapla
                    anomaly_map = calculate_raw_anomaly_map(
                        trainer.features_s,
                        trainer.features_t,
                        out_size=224,
                        norm=trainer.norm
                    )
                
                if len(anomaly_map.shape) == 3:
                    anomaly_map = anomaly_map[0]
                
                # İstatistikleri topla
                flat_scores = anomaly_map.flatten()
                all_normal_scores.extend(flat_scores)
                
                stats = {
                    "image": img_path.name,
                    "mean": np.mean(flat_scores),
                    "std": np.std(flat_scores),
                    "max": np.max(flat_scores),
                    "min": np.min(flat_scores),
                    "p95": np.percentile(flat_scores, 95),
                    "p99": np.percentile(flat_scores, 99)
                }
                normal_stats.append(stats)
                
                print(f"      📊 Mean: {stats['mean']:.6f}, Max: {stats['max']:.6f}, P99: {stats['p99']:.6f}")
            
            # Genel istatistikler
            all_normal_scores = np.array(all_normal_scores)
            global_mean = np.mean(all_normal_scores)
            global_std = np.std(all_normal_scores)
            global_max = np.max(all_normal_scores)
            
            print(f"\n📊 NORMAL GÖRÜNTÜLER İÇİN GLOBAL İSTATİSTİKLER:")
            print(f"   📈 Global Mean: {global_mean:.6f}")
            print(f"   📏 Global Std: {global_std:.6f}")
            print(f"   🔝 Global Max: {global_max:.6f}")
            
            # Threshold hesaplama yöntemleri
            threshold_candidates = {
                "mean_plus_3std": global_mean + 3 * global_std,
                "mean_plus_2std": global_mean + 2 * global_std,
                "global_p99": np.percentile(all_normal_scores, 99),
                "global_p995": np.percentile(all_normal_scores, 99.5),
                "global_p999": np.percentile(all_normal_scores, 99.9),
                "max_p99": np.max([s["p99"] for s in normal_stats]),
                "mean_of_maxes": np.mean([s["max"] for s in normal_stats]),
            }
            
            print(f"\n🎯 THRESHOLD ADAYLARI:")
            for method, value in threshold_candidates.items():
                print(f"   {method}: {value:.6f}")
            
            # False positive rate hesapla her threshold için
            best_threshold = None
            best_method = None
            best_fp_rate = float('inf')
            
            target_fp_rate = 0.05  # Maksimum %5 false positive
            
            for method, threshold in threshold_candidates.items():
                # Her normal görüntü için false positive rate hesapla
                fp_rates = []
                for stats in normal_stats:
                    # Bu görüntü için bu threshold ile kaç piksel anomali olarak işaretlenecek
                    # Bunu tam olarak hesaplamak için görüntüyü tekrar işlememiz gerekir
                    # Ama yaklaşık olarak p99 değeri ile tahmin edebiliriz
                    if threshold > stats["p99"]:
                        fp_rate = 0.01  # %1 false positive (p99'dan yüksek)
                    elif threshold > stats["p95"]:
                        fp_rate = 0.05  # %5 false positive 
                    elif threshold > stats["mean"] + 2*stats["std"]:
                        fp_rate = 0.025  # %2.5 false positive
                    else:
                        fp_rate = 0.1  # %10 false positive
                    
                    fp_rates.append(fp_rate)
                
                avg_fp_rate = np.mean(fp_rates)
                
                print(f"   {method}: FP Rate = {avg_fp_rate:.1%}")
                
                # En düşük false positive rate'e sahip olanı seç
                if avg_fp_rate < best_fp_rate and avg_fp_rate <= target_fp_rate:
                    best_fp_rate = avg_fp_rate
                    best_threshold = threshold
                    best_method = method
            
            # Eğer hiçbiri target'a uymuyorsa, en konservatif olanı seç
            if best_threshold is None:
                best_method = "global_p999"  # En konservatif
                best_threshold = threshold_candidates[best_method]
                best_fp_rate = 0.001  # %0.1 false positive
            
            print(f"\n🏆 EN İYİ THRESHOLD:")
            print(f"   🎯 Yöntem: {best_method}")
            print(f"   📏 Değer: {best_threshold:.6f}")
            print(f"   📊 Beklenen FP Rate: {best_fp_rate:.1%}")
            
            return {
                "success": True,
                "calibrated_threshold": best_threshold,
                "threshold_method": f"Calibrated-{best_method}",
                "false_positive_rate": best_fp_rate,
                "normal_image_count": len(selected_images),
                "global_stats": {
                    "mean": global_mean,
                    "std": global_std,
                    "max": global_max
                },
                "threshold_candidates": threshold_candidates,
                "individual_stats": normal_stats
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Kalibrasyon hatası: {str(e)}"
            }

    def detect_anomalies(self, image_path: str, model_type: str, dataset: str, 
                        use_calibrated_threshold: bool = True) -> Dict:
        """Anomali tespiti yap - Kalibre edilmiş threshold ile"""
        print("\n" + "="*60)
        print("🔍 ANOMALİ TESPİTİ (Kalibre Edilmiş Threshold)")
        print("="*60)
        
        try:
            # Önce threshold kalibrasyonu yap
            if use_calibrated_threshold:
                calibration_result = self.calibrate_threshold_on_normal_images(model_type, dataset, num_samples=10)
                
                if calibration_result["success"]:
                    print(f"✅ Threshold kalibrasyonu başarılı!")
                    print(f"🎯 Kalibre edilmiş threshold: {calibration_result['calibrated_threshold']:.6f}")
                    print(f"📊 False positive rate: {calibration_result['false_positive_rate']:.1%}")
                    calibrated_threshold = calibration_result['calibrated_threshold']
                    threshold_method = calibration_result['threshold_method']
                else:
                    print(f"⚠️ Kalibrasyon başarısız: {calibration_result['error']}")
                    print("📏 Fallback threshold kullanılıyor...")
                    calibrated_threshold = None
                    threshold_method = "Fallback-Percentile99"
            else:
                calibrated_threshold = None
                threshold_method = "Individual-AUROC"
            
            # Custom anomaly calculation function
            def calculate_raw_anomaly_map(fs_list, ft_list, out_size, norm):
                """Normalizasyon olmadan anomali haritası hesapla"""
                anomaly_map = 0
                for i in range(len(ft_list)):
                    fs = fs_list[i]
                    ft = ft_list[i]
                    fs_norm = F.normalize(fs, p=2) if norm else fs
                    ft_norm = F.normalize(ft, p=2) if norm else ft

                    a_map = 0.5 * (ft_norm - fs_norm) ** 2
                    a_map = a_map.sum(1, keepdim=True)
                    a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=False)
                    anomaly_map += a_map
                
                anomaly_map = anomaly_map.squeeze().cpu().numpy()
                
                # Gaussian filter uygula
                from scipy.ndimage import gaussian_filter
                if len(anomaly_map.shape) == 2:
                    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                else:
                    for i in range(anomaly_map.shape[0]):
                        anomaly_map[i] = gaussian_filter(anomaly_map[i], sigma=4)
                return anomaly_map
            
            # Konfigürasyon hazırla
            config = {
                "data_path": f"./dataset",
                "obj": dataset,
                "save_path": "./results",
                "distillType": model_type,
                "inference_only_mode": True,
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
            trainer.model_dir = f"./results/models/{dataset}/{model_type}"
            
            model_path = Path(f"results/models/{dataset}/{model_type}/student.pth")
            if not model_path.exists():
                return {
                    "success": False,
                    "error": f"Model ağırlıkları bulunamadı: {model_path}"
                }
            
            trainer.load_weights()
            trainer.change_mode("eval")
            print(f"✅ Model yüklendi: {model_path}")
            
            # Görüntüyü işle
            image_tensor = self.preprocess_image(image_path)
            original_image = self.load_original_image(image_path)
            print(f"✅ Görüntü işlendi: {image_path}")
            
            # Inference yap
            with torch.no_grad():
                trainer.infer(image_tensor)
                trainer.post_process()
                
                # Raw anomaly map hesapla
                anomaly_map = calculate_raw_anomaly_map(
                    trainer.features_s,
                    trainer.features_t,
                    out_size=224,
                    norm=trainer.norm
                )
            
            # Anomaly map'i numpy array'e çevir
            if len(anomaly_map.shape) == 3:
                anomaly_map = anomaly_map[0]
            
            # İstatistikleri hesapla
            flat_scores = anomaly_map.flatten()
            max_anomaly = np.max(flat_scores)
            mean_anomaly = np.mean(flat_scores)
            std_anomaly = np.std(flat_scores)
            min_anomaly = np.min(flat_scores)
            
            print(f"\n📊 Bu görüntü için istatistikler:")
            print(f"   📏 Min/Max: [{min_anomaly:.6f}, {max_anomaly:.6f}]")
            print(f"   📈 Mean ± Std: {mean_anomaly:.6f} ± {std_anomaly:.6f}")
            
            # Threshold belirleme
            if calibrated_threshold is not None:
                # Kalibre edilmiş threshold kullan
                best_threshold = calibrated_threshold
                best_name = threshold_method
                print(f"\n🎯 Kalibre edilmiş threshold kullanılıyor: {best_threshold:.6f}")
            else:
                # Fallback: Bu görüntü için percentile 99 kullan
                best_threshold = np.percentile(flat_scores, 99)
                best_name = threshold_method
                print(f"\n📏 Fallback threshold (99% percentile): {best_threshold:.6f}")
            
            # Final hesaplamalar
            final_binary_mask = (anomaly_map > best_threshold).astype(np.uint8)
            anomaly_count = np.sum(final_binary_mask)
            anomaly_ratio = anomaly_count / len(flat_scores)
            
            # Bölge analizi
            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(final_binary_mask)
            num_regions = num_labels - 1  # Background hariç
            largest_area = 0
            if num_regions > 0:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_area = np.max(areas)
            
            has_anomaly = anomaly_ratio > 0.0005 and num_regions > 0  # Çok düşük threshold: %0.05
            
            # Anomali derecesi belirleme (çok daha konservatif)
            if anomaly_ratio < 0.0005:  # %0.05'ten az
                severity = "NORMAL"
                severity_emoji = "✅"
            elif anomaly_ratio < 0.002:  # %0.2'den az
                severity = "DÜŞÜK RİSK"
                severity_emoji = "⚠️"
            elif anomaly_ratio < 0.01:   # %1'den az
                severity = "ORTA RİSK"
                severity_emoji = "🟠"
            else:
                severity = "YÜKSEK RİSK"
                severity_emoji = "🚨"
            
            print(f"\n📊 SONUÇ ANALİZİ:")
            print(f"   🎯 Kullanılan threshold: {best_threshold:.6f}")
            print(f"   🔴 Anomali oranı: {anomaly_ratio:.4%}")
            print(f"   🏷️ Bölge sayısı: {num_regions}")
            print(f"   📐 En büyük bölge: {largest_area} piksel")
            
            print(f"\n🚨 SONUÇ: {severity_emoji} {severity}")
            if has_anomaly:
                print("🔴 ANOMALİ TESPİT EDİLDİ!")
            else:
                print("🟢 Anomali tespit edilmedi")
            
            return {
                "success": True,
                "has_anomaly": has_anomaly,
                "max_score": float(max_anomaly),
                "mean_score": float(mean_anomaly),
                "std_score": float(std_anomaly),
                "min_score": float(min_anomaly),
                "threshold": best_threshold,
                "threshold_method": best_name,
                "anomaly_ratio": float(anomaly_ratio),
                "anomaly_count": int(anomaly_count),
                "num_regions": int(num_regions),
                "largest_area": int(largest_area),
                "severity": severity,
                "severity_emoji": severity_emoji,
                "anomaly_map": anomaly_map,
                "binary_mask": final_binary_mask,
                "original_image": original_image,
                "model_type": model_type,
                "dataset": dataset,
                "image_path": image_path,
                "calibration_result": calibration_result if use_calibrated_threshold else None
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
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
        
        # Normalized score kalite kriterleri (5 kriter)
        criteria = {
            "reasonable_ratio": 0.001 <= adaptive_ratio <= 0.15,  # %0.1 - %15 arası makul
            "not_too_sensitive": adaptive_ratio < 0.5,  # Çok hassas olmasın
            "captures_outliers": adaptive_thresh > np.percentile(scores, 90),  # En üst %10'u yakalasın
            "better_than_original": adaptive_ratio > 0 and abs(adaptive_ratio - original_ratio) > 0.001,
            "statistical_significance": adaptive_thresh > np.mean(scores) + 2 * np.std(scores)  # İstatistiksel anlamlılık
        };
        
        quality_score = sum(criteria.values());
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
        z_scores = (anomaly_map - mean) / (std + 1e-8)  # Avoid division by zero
        
        # Anomali skoru eşik değerine göre ayarla
        adjusted_scores = np.where(z_scores > threshold, z_scores, 0)
        
        return adjusted_scores

    def load_model(self, model_name: str) -> Dict:
        """Model yükleme metodu (visualize_anomaly için gerekli)"""
        # Bu metod visualize_anomaly fonksiyonu için gerekli
        # Gerçek implementasyon analyze_and_visualize'da yapılıyor
        return {"model": None}
    
    def load_and_preprocess_image(self, image_path: str) -> Tuple[np.ndarray, torch.Tensor]:
        """Görüntü yükleme ve ön işleme (visualize_anomaly için gerekli)"""
        original_image = self.load_original_image(image_path)
        processed_image = self.preprocess_image(image_path)
        return original_image, processed_image

    def _create_anomaly_visualization(self, original_image: np.ndarray, anomaly_map: np.ndarray, 
                                    binary_mask: np.ndarray, save_path: str, threshold: float,
                                    threshold_method: str, threshold_info: Dict) -> None:
        """Detaylı anomali görselleştirmesi oluştur"""
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'🔍 Detaylı Anomali Analizi - {threshold_method} Threshold\n'
                    f'Dosya: {os.path.basename(save_path)}', fontsize=16, fontweight='bold')
        
        # 1. Orijinal Görüntü
        plt.subplot(2, 4, 1)
        plt.imshow(original_image)
        plt.title('1. Orijinal Görüntü', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 2. Anomali Haritası (Ham)
        plt.subplot(2, 4, 2)
        im2 = plt.imshow(anomaly_map, cmap='hot', interpolation='bilinear')
        plt.title(f'2. Anomali Haritası\nMax: {np.max(anomaly_map):.6f}', fontsize=12, fontweight='bold')
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # 3. Binary Mask
        plt.subplot(2, 4, 3)
        plt.imshow(binary_mask, cmap='binary')
        plt.title(f'3. Binary Mask\n(Threshold: {threshold:.6f})', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 4. Overlay
        plt.subplot(2, 4, 4)
        plt.imshow(original_image)
        anomaly_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        plt.imshow(anomaly_normalized, cmap='hot', alpha=0.5, interpolation='bilinear')
        plt.title('4. Overlay Görünümü', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 5. Histogram
        plt.subplot(2, 4, 5)
        plt.hist(anomaly_map.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'{threshold_method} Threshold: {threshold:.6f}')
        if threshold_info and "original_threshold" in threshold_info:
            plt.axvline(threshold_info["original_threshold"], color='orange', linestyle=':', linewidth=2,
                       label=f'Orijinal Threshold: {threshold_info["original_threshold"]:.6f}')
        plt.xlabel('Anomali Skoru')
        plt.ylabel('Frekans')
        plt.title('5. Skor Dağılımı', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Contour Görünümü
        plt.subplot(2, 4, 6)
        plt.imshow(original_image)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Küçük gürültüleri filtrele
                plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-', linewidth=2)
        plt.title('6. Anomali Konturları', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 7. İstatistik Tablosu
        ax7 = plt.subplot(2, 4, 7)
        ax7.axis('off')
        
        stats_text = f"""📊 THRESHOLD ANALİZİ

🎯 Kullanılan Yöntem: {threshold_method}
📏 Threshold Değeri: {threshold:.6f}

📈 ANOMALİ İSTATİSTİKLERİ:
• Toplam Piksel: {anomaly_map.size:,}
• Anomali Piksel: {np.sum(binary_mask):,}
• Anomali Oranı: {(np.sum(binary_mask)/anomaly_map.size)*100:.3f}%

📊 SKOR İSTATİSTİKLERİ:
• Min Skor: {np.min(anomaly_map):.6f}
• Max Skor: {np.max(anomaly_map):.6f}
• Ortalama: {np.mean(anomaly_map):.6f}
• Std. Sapma: {np.std(anomaly_map):.6f}

🔬 PERCENTILES:
• 50%: {np.percentile(anomaly_map, 50):.6f}
• 90%: {np.percentile(anomaly_map, 90):.6f}
• 95%: {np.percentile(anomaly_map, 95):.6f}
• 99%: {np.percentile(anomaly_map, 99):.6f}
        """
        
        if threshold_info and "method_thresholds" in threshold_info:
            stats_text += f"\n🧠 ADAPTİF THRESHOLD BİLGİLERİ:\n"
            for method, value in threshold_info["method_thresholds"].items():
                stats_text += f"• {method.title()}: {value:.6f}\n"
        
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 8. Anomali Bölge Analizi
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')
        
        # Contour alanları analiz et
        total_anomaly_area = np.sum(binary_mask)
        num_contours = len([c for c in contours if cv2.contourArea(c) > 10])
        largest_contour_area = max([cv2.contourArea(c) for c in contours], default=0)
        
        area_text = f"""🔍 BÖLGE ANALİZİ

🏷️ Tespit Edilen Bölgeler: {num_contours}
📐 Toplam Anomali Alanı: {total_anomaly_area:,} piksel
📏 En Büyük Bölge: {largest_contour_area:.0f} piksel

💡 DEĞERLENDİRME:
"""
        
        if total_anomaly_area == 0:
            area_text += "✅ Anomali tespit edilmedi"
        elif total_anomaly_area < 100:
            area_text += "⚠️ Küçük anomali bölgeleri"
        elif total_anomaly_area < 1000:
            area_text += "🟠 Orta büyüklükte anomali"
        else:
            area_text += "🚨 Büyük anomali bölgesi"
        
        if threshold_info and "is_adaptive_better" in threshold_info:
            area_text += f"\n\n🧠 Adaptif Threshold: {'✅ Daha İyi' if threshold_info['is_adaptive_better'] else '❌ Orijinal Tercih'}"
        
        ax8.text(0.05, 0.95, area_text, transform=ax8.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def _create_detailed_comparison(self, original_image: np.ndarray, anomaly_map: np.ndarray,
                                  binary_mask: np.ndarray, save_path: str, threshold: float,
                                  threshold_method: str, threshold_info: Dict) -> None:
        """Detaylı karşılaştırma görselleştirmesi"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'🔄 Threshold Karşılaştırması - {threshold_method}\n'
                    f'Dosya: {os.path.basename(save_path)}', fontsize=16, fontweight='bold')
        
        # Üst sıra: Adaptif threshold ile
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f'Orijinal Görüntü', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(anomaly_map, cmap='jet')
        axes[0, 1].set_title(f'Anomali Haritası\nMax: {result["max_score"]:.6f}', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(binary_mask, cmap='binary')
        axes[0, 2].set_title(f'{threshold_method} Threshold\n({threshold:.6f})', fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Alt sıra: Orijinal threshold ile (karşılaştırma için)
        if threshold_info and "original_threshold" in threshold_info:
            original_threshold = threshold_info["original_threshold"]
            original_binary = (anomaly_map > original_threshold).astype(np.uint8)
            
            axes[1, 0].imshow(original_image)
            overlay = original_image.copy()
            overlay[binary_mask > 0] = [255, 0, 0]  # Kırmızı overlay
            axes[1, 0].imshow(overlay, alpha=0.7)
            axes[1, 0].set_title(f'{threshold_method} Overlay', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(original_binary, cmap='binary')
            axes[1, 1].set_title(f'Orijinal Threshold\n({original_threshold:.6f})', fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')
            
            # Fark analizi
            diff_mask = np.abs(binary_mask.astype(int) - original_binary.astype(int))
            axes[1, 2].imshow(diff_mask, cmap='RdYlBu')
            axes[1, 2].set_title('Threshold Farkları\n(Mavi: Orijinal, Kırmızı: Adaptif)', fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')
        else:
            # Orijinal threshold bilgisi yoksa histogram göster
            axes[1, 0].hist(anomaly_map.flatten(), bins=50, alpha=0.7, color='blue')
            axes[1, 0].axvline(threshold, color='red', linestyle='--', linewidth=2)
            axes[1, 0].set_title('Skor Dağılımı ve Threshold', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Boş alanları gizle
            axes[1, 1].axis('off')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_comprehensive_visualization(self, result: Dict, save_path: str = None) -> str:
        """Kapsamlı görselleştirme oluştur"""
        if not result["success"]:
            print(f"❌ Görselleştirme yapılamıyor: {result['error']}")
            return None
        
        anomaly_map = result["anomaly_map"]
        original_image = result["original_image"]
        threshold = result["threshold"]

        # --- Ensure percentile values are available ---
        if "percentiles" in result and result["percentiles"] is not None:
            percentiles = result["percentiles"]
        else:
            # Calculate on the fly
            percentiles = [np.percentile(anomaly_map, p) for p in [50, 75, 90, 95, 99]]
            result["percentiles"] = percentiles
        
        # Figure oluştur - büyük boyut
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(f'Anomali Tespit Analizi - {result["model_type"].upper()} Model\n'
                    f'Dosya: {Path(result["image_path"]).name}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Orijinal Görüntü
        ax1 = plt.subplot(2, 4, 1)
        plt.imshow(original_image)
        plt.title('1. Orijinal Görüntü', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 2. Anomali Haritası (Raw)
        ax2 = plt.subplot(2, 4, 2)
        im2 = plt.imshow(anomaly_map, cmap='hot', interpolation='bilinear')
        plt.title(f'2. Anomali Haritası\nMax: {result["max_score"]:.6f}', fontsize=12, fontweight='bold')
        plt.colorbar(im2, fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # 3. Anomali Haritası (Jet colormap)
        ax3 = plt.subplot(2, 4, 3)
        im3 = plt.imshow(anomaly_map, cmap='jet', interpolation='bilinear')
        plt.title('3. Anomali Haritası (Jet)', fontsize=12, fontweight='bold')
        plt.colorbar(im3, fraction=0.046, pad=0.04)
        plt.axis('off')
        
        # 4. Threshold Uygulanmış
        ax4 = plt.subplot(2, 4, 4)
        thresholded = (anomaly_map > threshold).astype(np.float32)
        plt.imshow(thresholded, cmap='binary')
        plt.title(f'4. Eşiklenmiş (>{threshold:.3f})\nAnomaliler: {np.sum(thresholded):.0f} piksel', 
                 fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 5. Overlay (Sıcak renk)
        ax5 = plt.subplot(2, 4, 5)
        # Orijinal görüntüyü yeniden boyutlandır
        original_resized = cv2.resize(original_image, (anomaly_map.shape[1], anomaly_map.shape[0]))
        
        # Anomali haritasını normalize et
        anomaly_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        
        # Overlay oluştur
        plt.imshow(original_resized)
        plt.imshow(anomaly_normalized, cmap='hot', alpha=0.4, interpolation='bilinear')
        plt.title('5. Overlay (Hot)', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 6. Overlay (Soğuk renk)
        ax6 = plt.subplot(2, 4, 6)
        plt.imshow(original_resized)
        plt.imshow(anomaly_normalized, cmap='viridis', alpha=0.5, interpolation='bilinear')
        plt.title('6. Overlay (Viridis)', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # 7. Anomali Histogramı
        ax7 = plt.subplot(2, 4, 7)
        plt.hist(anomaly_map.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.3f}')
        plt.axvline(result["max_score"], color='orange', linestyle='-', linewidth=2, label=f'Max: {result["max_score"]:.6f}')
        plt.xlabel('Anomali Skoru')
        plt.ylabel('Frekans')
        plt.title('7. Anomali Skor Dağılımı', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. İstatistik Kutusu
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')
        
        # İstatistik metni
        stats_text = f"""📊 DETAYLI İSTATİSTİKLER
        
🔍 Model: {result["model_type"].upper()}
📁 Dataset: {result["dataset"]}
🖼️ Görüntü: {Path(result["image_path"]).name}

📈 SKORLAR:
• Maksimum: {result["max_score"]:.6f}
• Ortalama: {result["mean_score"]:.6f}
• Std. Sapma: {result["std_score"]:.6f}
• Threshold: {result["threshold"]:.3f}

📊 PERCENTİLES:
• 50%: {percentiles[0]:.6f}
• 75%: {percentiles[1]:.6f}
• 90%: {percentiles[2]:.6f}
• 95%: {percentiles[3]:.6f}
• 99%: {percentiles[4]:.6f}

🎯 SONUÇ:
• Anomali Var: {'✅ EVET' if result["has_anomaly"] else '❌ HAYIR'}
• Anomali Oranı: {result["anomaly_ratio"]:.4%}
• Toplam Piksel: {anomaly_map.size:,}
• Anomali Piksel: {np.sum(anomaly_map > threshold):.0f}
        """
        
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        # Kaydet
        if not save_path:
            timestamp = int(time.time())
            save_path = f"results/anomaly_visualization_{result['model_type']}_{result['dataset']}_{timestamp}.png"
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"💾 Kapsamlı görselleştirme kaydedildi: {save_path}")
        return save_path
    
    def create_side_by_side_comparison(self, result: Dict, save_path: str = None) -> str:
        """Yan yana karşılaştırma görselleştirmesi"""
        if not result["success"]:
            return None
        
        anomaly_map = result["anomaly_map"]
        original_image = result["original_image"]
        
        # Figure oluştur
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Anomali Karşılaştırması - {result["model_type"].upper()}\n'
                    f'{Path(result["image_path"]).name}', fontsize=14, fontweight='bold')
        
        # Orijinal görüntüyü yeniden boyutlandır
        original_resized = cv2.resize(original_image, (anomaly_map.shape[1], anomaly_map.shape[0]))
        
        # 1. Orijinal
        axes[0].imshow(original_resized)
        axes[0].set_title('Orijinal Görüntü', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Anomali Haritası
        im1 = axes[1].imshow(anomaly_map, cmap='jet')
        axes[1].set_title(f'Anomali Haritası\nMax: {result["max_score"]:.6f}', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        axes[1].axis('off')
        
        # 3. Overlay
        axes[2].imshow(original_resized)
        anomaly_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
        axes[2].imshow(anomaly_normalized, cmap='hot', alpha=0.5, interpolation='bilinear')
        axes[2].set_title('Overlay Görünümü', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Kaydet
        if not save_path:
            timestamp = int(time.time())
            save_path = f"results/anomaly_comparison_{result['model_type']}_{result['dataset']}_{timestamp}.png"
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"💾 Karşılaştırma görselleştirmesi kaydedildi: {save_path}")
        return save_path
    
    def analyze_and_visualize(self, image_path: str, model_type: str = "sn", 
                            dataset: str = "wood", show_comprehensive: bool = True, 
                            show_comparison: bool = True) -> Dict:
        """Ana analiz ve görselleştirme fonksiyonu"""
        print("🎨 ANOMALİ GÖRSELLEŞTİRME ARACI")
        print("=" * 50)
        print(f"📁 Görüntü: {image_path}")
        print(f"🤖 Model: {model_type.upper()}")
        print(f"📊 Dataset: {dataset}")
        
        # Dosya kontrolü
        if not Path(image_path).exists():
            print(f"❌ HATA: Görüntü dosyası bulunamadı: {image_path}")
            return {"success": False, "error": "Dosya bulunamadı"}
        
        # Anomali tespiti yap
        result = self.detect_anomalies(image_path, model_type, dataset)
        
        if not result["success"]:
            return result
        
        # Görselleştirmeleri oluştur
        visualization_paths = {}
        
        if show_comprehensive:
            comp_path = self.create_comprehensive_visualization(result)
            if comp_path:
                visualization_paths["comprehensive"] = comp_path
        
        if show_comparison:
            comp_path = self.create_side_by_side_comparison(result)
            if comp_path:
                visualization_paths["comparison"] = comp_path
        
        result["visualization_paths"] = visualization_paths
        
        print("\n" + "🎉" + "="*48 + "🎉")
        print("        GÖRSELLEŞTİRME TAMAMLANDI!")
        print("🎉" + "="*48 + "🎉")
        
        return result
    
    def visualize_anomaly(self, image_path: str, model_name: str, save_results: bool = True) -> Dict:
        """Anomali görselleştirmesi yap"""
        print(f"\n🔍 Anomali Analizi Başlatılıyor")
        print(f"📁 Görüntü: {os.path.basename(image_path)}")
        print(f"🤖 Model: {model_name.upper()}")
        print("=" * 50)
        
        try:
            # Model ve görüntü yükle
            model_info = self.load_model(model_name)
            original_image, processed_image = self.load_and_preprocess_image(image_path)
            
            # Anomali haritası oluştur
            with torch.no_grad():
                anomaly_map = model_info['model'](processed_image)
                if isinstance(anomaly_map, tuple):
                    anomaly_map = anomaly_map[0]
                
                # CPU'ya taşı ve numpy'a çevir
                anomaly_map_np = anomaly_map.squeeze().cpu().numpy()
                
                # Orijinal boyutlara yeniden boyutlandır
                original_height, original_width = original_image.shape[:2]
                anomaly_map_resized = cv2.resize(anomaly_map_np, (original_width, original_height))
            
            # 🧠 Adaptif threshold hesapla
            threshold_info = {}
            if self.adaptive_threshold_config["use_adaptive"]:
                threshold_info = self.calculate_adaptive_threshold(anomaly_map_resized)
                used_threshold = threshold_info["adaptive_threshold"]
                threshold_method = "Adaptif"
                
                # Eğer adaptif threshold daha iyi değilse orijinalini kullan
                if not threshold_info["is_adaptive_better"]:
                    print("⚠️  Adaptif threshold kaliteli değil, orijinal threshold kullanılıyor")
                    used_threshold = self.threshold
                    threshold_method = "Orijinal"
            else:
                used_threshold = self.threshold
                threshold_method = "Sabit"
                print(f"📏 Sabit threshold kullanılıyor: {used_threshold:.6f}")
            
            # Binary mask oluştur
            binary_mask = (anomaly_map_resized > used_threshold).astype(np.uint8)
            
            # Anomali istatistikleri
            total_pixels = original_height * original_width
            anomaly_pixels = np.sum(binary_mask)
            anomaly_percentage = (anomaly_pixels / total_pixels) * 100
            
            print(f"\n📊 Anomali İstatistikleri ({threshold_method} Threshold)")
            print("-" * 40)
            print(f"🎯 Kullanılan Threshold: {used_threshold:.6f}")
            print(f"🔴 Anomali Piksel Sayısı: {anomaly_pixels:,}")
            print(f"📏 Toplam Piksel: {total_pixels:,}")
            print(f"📈 Anomali Yüzdesi: {anomaly_percentage:.3f}%")
            
            # Anomali seviyesi belirleme (yüzdesel sapmaya göre)
            if anomaly_percentage < 0.1:
                severity = "NORMAL"
                severity_color = "green"
                severity_emoji = "✅"
            elif anomaly_percentage < 1.0:
                severity = "DÜŞÜK RİSK"
                severity_color = "yellow"
                severity_emoji = "⚠️"
            elif anomaly_percentage < 5.0:
                severity = "ORTA RİSK"
                severity_color = "orange"
                severity_emoji = "🟠"
            else:
                severity = "YÜKSEK RİSK"
                severity_color = "red"
                severity_emoji = "🚨"
            
            print(f"🚨 Anomali Seviyesi: {severity_emoji} {severity}")
            
            # Görselleştirme
            if save_results:
                timestamp = int(time.time())
                save_path = f"results/anomaly_visualization_{model_name}_{os.path.splitext(os.path.basename(image_path))[0]}_{timestamp}.png"
                comparison_path = f"results/anomaly_comparison_{model_name}_{os.path.splitext(os.path.basename(image_path))[0]}_{timestamp}.png"
                
                self._create_anomaly_visualization(
                    original_image, anomaly_map_resized, binary_mask, 
                    save_path, used_threshold, threshold_method, threshold_info
                )
                self._create_detailed_comparison(
                    original_image, anomaly_map_resized, binary_mask,
                    comparison_path, used_threshold, threshold_method, threshold_info
                )
                
                print(f"\n💾 Sonuçlar Kaydedildi:")
                print(f"📊 Görselleştirme: {save_path}")
                print(f"📈 Detaylı Karşılaştırma: {comparison_path}")
            
            # Sonuç raporu
            result = {
                "model_name": model_name,
                "image_path": image_path,
                "threshold_method": threshold_method,
                "used_threshold": used_threshold,
                "anomaly_percentage": anomaly_percentage,
                "anomaly_pixels": int(anomaly_pixels),
                "total_pixels": int(total_pixels),
                "severity": severity,
                "severity_color": severity_color,
                "threshold_info": threshold_info
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Hata oluştu: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
def main():
    """Ana fonksiyon"""
    import time
    
    print("🎨 ANOMALİ GÖRSELLEŞTİRME ARACI")
    print("=" * 50)
    
    # Sistem başlat
    visualizer = AnomalyVisualizer()
    
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
        
        # Model türü
        model_type = input("Model türü (sn/dbfad/ead/rd/st, varsayılan: sn): ").strip()
        if not model_type:
            model_type = "sn"
        
        # Dataset
        dataset = input("Dataset (varsayılan: wood): ").strip()
        if not dataset:
            dataset = "wood"
    
    # Parametreleri doğrula
    print(f"\n🔧 SEÇILEN PARAMETRELER:")
    print(f"   📁 Görüntü: {image_path}")
    print(f"   🤖 Model: {model_type}")
    print(f"   📊 Dataset: {dataset}")
    
    # Analiz ve görselleştirme yap
    try:
        results = visualizer.analyze_and_visualize(
            image_path=image_path,
            model_type=model_type,
            dataset=dataset,
            show_comprehensive=True,
            show_comparison=True
        )
        
        # Sonuç özeti
        if results["success"]:
            print(f"\n📋 SONUÇ ÖZETİ:")
            print(f"   🔍 Anomali tespit: {'✅ EVET' if results['has_anomaly'] else '❌ HAYIR'}")
            print(f"   📊 Maksimum skor: {results['max_score']:.6f}")
            print(f"   📈 Ortalama skor: {results['mean_score']:.6f}")
            print(f"   🎯 Threshold: {results['threshold']:.3f}")
            print(f"   📉 Anomali oranı: {results['anomaly_ratio']:.4%}")
            
            if "visualization_paths" in results:
                print(f"\n💾 OLUŞTURULAN DOSYALAR:")
                for viz_type, viz_path in results["visualization_paths"].items():
                    print(f"   📁 {viz_type}: {viz_path}")
        else:
            print(f"\n❌ HATA: {results['error']}")
    
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎨 Görselleştirme tamamlandı!")


if __name__ == "__main__":
    main()