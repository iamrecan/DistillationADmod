import torch
import cv2
import numpy as np
import sys
import os
from segment_anything import build_sam, SamPredictor
import torch.nn as nn
from typing import List, Tuple, Union

class SAM2Segmenter:
    def __init__(self, 
                 model_path=None, 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 img_size: int = 1024,
                 multimask_output: bool = True,
                 confidence_threshold: float = 0.3):  # Eşik değerini düşürdük
        """
        SAM2 segmentasyon modeli için başlatıcı.
        """
        self.device = device
        self.img_size = img_size
        self.multimask_output = multimask_output
        self.confidence_threshold = confidence_threshold
        
        # Model yolunu belirle
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "sam2.1_hiera_small.pt")
        
        print(f"Model yükleniyor: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        # Model state dict'i yükle
        try:
            self.sam = build_sam()
            print("SAM modeli oluşturuldu")
            
            state_dict = torch.load(model_path, map_location=device)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            
            # Uyumlu olmayan anahtarları temizle
            compatible_state_dict = {}
            for key, value in state_dict.items():
                if key in self.sam.state_dict():
                    compatible_state_dict[key] = value
            
            self.sam.load_state_dict(compatible_state_dict, strict=False)
            print("Model ağırlıkları yüklendi")
            
            self.sam.to(device)
            self.predictor = SamPredictor(self.sam)
            print("Model hazır")
            
        except Exception as e:
            print(f"Model yüklenirken hata: {str(e)}")
            raise e
    
    def check_point_validity(self, point: Tuple[int, int], image_shape: Tuple[int, int]) -> bool:
        """Noktanın görüntü sınırları içinde olup olmadığını kontrol et"""
        h, w = image_shape[:2]
        x, y = point
        return 0 <= x < w and 0 <= y < h
    
    def adjust_point(self, point: Tuple[int, int], image_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Noktayı görüntü sınırları içine al"""
        h, w = image_shape[:2]
        x = max(0, min(point[0], w-1))
        y = max(0, min(point[1], h-1))
        return (x, y)
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Görüntüyü ön işleme"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Orijinal boyutları kaydet
        original_size = (image.shape[0], image.shape[1])
        
        # En-boy oranını koru
        h, w = original_size
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
        
        # Padding ekle
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        return image, original_size
    
    def scale_points(self, points: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """Noktaları model boyutuna ölçeklendir"""
        # Önce noktaları görüntü sınırları içine al
        h, w = original_size
        points = np.clip(points, [0, 0], [w-1, h-1])
        
        # Ölçeklendirme faktörlerini hesapla
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Padding hesapla
        pad_h = self.img_size - new_h
        pad_w = self.img_size - new_w
        top = pad_h // 2
        left = pad_w // 2
        
        # Noktaları ölçeklendir
        scaled_points = points.copy()
        scaled_points[:, 0] = points[:, 0] * scale + left
        scaled_points[:, 1] = points[:, 1] * scale + top
        
        return scaled_points.astype(int)
    
    def segment_from_points(self, 
                          image: np.ndarray, 
                          points: np.ndarray,
                          point_labels: np.ndarray = None,
                          ) -> Tuple[np.ndarray, float]:
        """Birden fazla noktadan segmentasyon gerçekleştirir"""
        try:
            # Görüntü boyutlarını kontrol et
            h, w = image.shape[:2]
            print(f"Görüntü boyutları: {w}x{h}")
            
            # Noktaları kontrol et ve düzelt
            points = np.clip(points, [0, 0], [w-1, h-1])
            
            # Görüntü ön işleme
            resized_image, original_size = self.preprocess_image(image)
            
            # Nokta etiketlerini hazırla
            if point_labels is None:
                point_labels = np.ones(len(points), dtype=int)
            
            # Noktaları ölçeklendir
            scaled_points = self.scale_points(points, original_size)
            
            # Görüntüyü SAM predictor'a set et
            self.predictor.set_image(resized_image)
            print(f"İşlenen görüntü boyutu: {resized_image.shape}")
            print(f"Nokta sayısı: {len(points)}")
            
            # Noktaların ortalama pozisyonunu hesapla ve göster
            mean_point = points.mean(axis=0).astype(int)
            print(f"Noktaların ortalama pozisyonu: ({mean_point[0]}, {mean_point[1]})")
            print(f"Ölçeklendirilmiş ortalama pozisyon: ({scaled_points.mean(axis=0)[0]:.1f}, {scaled_points.mean(axis=0)[1]:.1f})")
            
            # Segmentasyonu gerçekleştir
            masks, scores, logits = self.predictor.predict(
                point_coords=scaled_points,
                point_labels=point_labels,
                multimask_output=self.multimask_output
            )
            
            # En yüksek skorlu maskeyi seç
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            score = scores[best_mask_idx]
            
            print(f"Segmentasyon skoru: {score:.3f}")
            if score < self.confidence_threshold:
                print(f"Uyarı: Düşük güven skoru ({score:.3f} < {self.confidence_threshold})")
            
            # Maskeyi orijinal boyuta ölçeklendir
            best_mask = cv2.resize(
                best_mask.astype(np.uint8), 
                (original_size[1], original_size[0]), 
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            # Maskeyi temizle ve düzgünleştir
            if score > 0.1:  # Sadece makul skorlarda morfolojik işlemler uygula
                kernel = np.ones((3,3), np.uint8)  # Kernel boyutunu küçülttük
                best_mask = cv2.morphologyEx(best_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel)
                best_mask = best_mask.astype(bool)
            
            return best_mask, score
        
        except Exception as e:
            print(f"Segmentasyon sırasında hata: {str(e)}")
            return None, 0.0

def generate_random_points_around_center(center_x: int, 
                                      center_y: int, 
                                      num_points: int = 30, 
                                      min_radius: int = 5,
                                      max_radius: int = 15,
                                      image_shape: Tuple[int, int] = None) -> np.ndarray:
    """
    Merkez nokta etrafında rastgele noktalar üretir
    
    Args:
        center_x: Merkez noktanın x koordinatı
        center_y: Merkez noktanın y koordinatı
        num_points: Üretilecek nokta sayısı
        min_radius: Minimum yarıçap
        max_radius: Maksimum yarıçap
        image_shape: Görüntü boyutları (height, width)
    """
    # Rastgele açılar üret - uniform dağılım yerine normal dağılım kullan
    angles = np.random.normal(0, np.pi/2, num_points) % (2*np.pi)
    
    # Yarıçaplar için normal dağılım kullan
    mean_radius = (min_radius + max_radius) / 2
    std_radius = (max_radius - min_radius) / 4
    distances = np.clip(
        np.random.normal(mean_radius, std_radius, num_points),
        min_radius,
        max_radius
    )
    
    # Polar koordinatları kartezyen koordinatlara dönüştür
    x = center_x + distances * np.cos(angles)
    y = center_y + distances * np.sin(angles)
    
    # Noktaları birleştir
    points = np.column_stack([x, y])
    
    # Görüntü sınırları verilmişse, noktaları sınırlar içinde tut
    if image_shape is not None:
        h, w = image_shape
        points[:, 0] = np.clip(points[:, 0], 0, w-1)
        points[:, 1] = np.clip(points[:, 1], 0, h-1)
    
    return points.astype(int)

def main():
    if len(sys.argv) != 4:
        print("Kullanım: python sam2_for_anomaly_point.py <image_path> <x_coordinate> <y_coordinate>")
        return
    
    image_path = sys.argv[1]
    center_x = int(sys.argv[2])
    center_y = int(sys.argv[3])
    
    try:
        print("SAM2 modelini yüklüyor...")
        segmenter = SAM2Segmenter(
            img_size=1024,
            multimask_output=True,
            confidence_threshold=0.3  # Eşik değerini düşürdük
        )
        
        print(f"Görüntü yükleniyor: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Görüntü yüklenemedi: {image_path}")
            return
        
        # Merkez noktayı görüntü sınırları içine al
        h, w = image.shape[:2]
        center_x = max(0, min(center_x, w-1))
        center_y = max(0, min(center_y, h-1))
        
        print(f"Düzeltilmiş merkez nokta: ({center_x}, {center_y})")
        
        # Merkez nokta etrafında rastgele noktalar üret
        points = generate_random_points_around_center(
            center_x=center_x,
            center_y=center_y,
            num_points=30,
            min_radius=5,
            max_radius=15,
            image_shape=image.shape[:2]
        )
            
        print("Segmentasyon başlıyor...")
        mask, score = segmenter.segment_from_points(
            image, 
            points,
            point_labels=None
        )
        
        if mask is None:
            print("Segmentasyon başarısız oldu")
            return
            
        # Sonucu görselleştir
        masked_image = image.copy()
        overlay = np.zeros_like(image)
        overlay[mask] = [0, 255, 0]
        
        # Alfa değerini düşür
        masked_image = cv2.addWeighted(masked_image, 0.8, overlay, 0.2, 0)
        
        # Noktaları göster
        for point in points:
            cv2.circle(masked_image, tuple(point), 2, (0, 0, 255), -1)
        
        # Ortalama noktayı göster
        mean_point = points.mean(axis=0).astype(int)
        cv2.circle(masked_image, tuple(mean_point), 5, (255, 0, 0), -1)
        
        # Skoru ve diğer bilgileri görüntüye ekle
        info_text = [
            f"Score: {score:.3f}",
            f"Points: {len(points)}",
            f"Center: ({center_x}, {center_y})"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(
                masked_image,
                text,
                (10, 30 + i*30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        
        # Sonuçları göster
        cv2.imshow("Original", image)
        cv2.imshow("Segmented", masked_image)
        
        print("Görüntüler gösteriliyor. Çıkmak için herhangi bir tuşa basın...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Sonucu kaydet
        output_path = image_path.replace('.png', '_segmented_multipoint.png')
        cv2.imwrite(output_path, masked_image)
        print(f"Segmente edilmiş görüntü kaydedildi: {output_path}")
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()