#!/usr/bin/env python3
"""
🎨 ANOMALİ GÖRSELLEŞTIRME ARACI
Anomali tespit sonuçlarını detaylı şekilde görselleştir ve karşılaştır
"""

import torch
import numpy as np
import cv2
import os
import sys
import time
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as transforms
from typing import Dict, List, Optional, Tuple
import seaborn as sns

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
        self.threshold = 0.291  # Varsayılan threshold
        
        print(f"🎨 Anomali Görselleştirici başlatıldı - Cihaz: {self.device}")
        
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
    
    def detect_anomalies(self, image_path: str, model_type: str, dataset: str) -> Dict:
        """Anomali tespiti yap ve sonuçları döndür"""
        print("\n" + "="*60)
        print("🔍 ANOMALİ TESPİTİ VE GÖRSELLEŞTİRME")
        print("="*60)
        
        try:
            # Import the anomaly calculation function
            from utils.functions import cal_anomaly_maps
            
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
                
                # Anomaly map hesapla
                anomaly_map = cal_anomaly_maps(
                    trainer.features_s,
                    trainer.features_t,
                    out_size=224,
                    norm=trainer.norm
                )
            
            # Anomaly map'i numpy array'e çevir
            if len(anomaly_map.shape) == 3:
                anomaly_map = anomaly_map[0]
            
            # İstatistikleri hesapla
            max_anomaly = np.max(anomaly_map)
            mean_anomaly = np.mean(anomaly_map)
            std_anomaly = np.std(anomaly_map)
            anomaly_pixels = np.sum(anomaly_map > self.threshold)
            total_pixels = anomaly_map.size
            anomaly_ratio = anomaly_pixels / total_pixels
            
            # Percentile'ları hesapla
            percentiles = np.percentile(anomaly_map, [50, 75, 90, 95, 99])
            
            print(f"📊 Anomali İstatistikleri:")
            print(f"   Maksimum skor: {max_anomaly:.6f}")
            print(f"   Ortalama skor: {mean_anomaly:.6f}")
            print(f"   Standart sapma: {std_anomaly:.6f}")
            print(f"   Threshold: {self.threshold}")
            print(f"   Anomali pikselleri: {anomaly_pixels}/{total_pixels} ({anomaly_ratio:.2%})")
            print(f"   Percentiles [50,75,90,95,99]: {percentiles}")
            
            # Anomali var mı?
            has_anomaly = max_anomaly > self.threshold and anomaly_ratio > 0.001
            
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
                "threshold": self.threshold,
                "anomaly_ratio": float(anomaly_ratio),
                "percentiles": percentiles.tolist(),
                "anomaly_map": anomaly_map,
                "original_image": original_image,
                "model_type": model_type,
                "dataset": dataset,
                "image_path": image_path
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": f"Anomali tespit hatası: {str(e)}"}
    
    def create_comprehensive_visualization(self, result: Dict, save_path: str = None) -> str:
        """Kapsamlı görselleştirme oluştur"""
        if not result["success"]:
            print(f"❌ Görselleştirme yapılamıyor: {result['error']}")
            return None
        
        anomaly_map = result["anomaly_map"]
        original_image = result["original_image"]
        threshold = result["threshold"]
        
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
• 50%: {result["percentiles"][0]:.6f}
• 75%: {result["percentiles"][1]:.6f}
• 90%: {result["percentiles"][2]:.6f}
• 95%: {result["percentiles"][3]:.6f}
• 99%: {result["percentiles"][4]:.6f}

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