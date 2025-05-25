#!/usr/bin/env python3
"""
SAM2 (Segment Anything Model 2) Minimal Örnek Script
Bu script SAM2 modelini kullanarak resim segmentasyonu yapar.
Sadece numpy ve torch kullanır.
"""

import os
import sys
import numpy as np
import torch

# SAM2 modüllerini import et
sys.path.append('./sam2')

def main():
    """Ana fonksiyon"""
    print("SAM2 Minimal Örnek Script")
    print("=" * 50)
    
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        print("SAM2 modülleri başarıyla import edildi!")
    except Exception as e:
        print(f"SAM2 import hatası: {e}")
        return
    
    # Device ayarları
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA kullanılıyor")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple MPS kullanılıyor")
    else:
        device = torch.device("cpu")
        print("CPU kullanılıyor")
    
    print(f"Device: {device}")
    
    # MPS için fallback ayarı
    if device.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("MPS fallback etkinleştirildi")
    
    # Model dosyalarını kontrol et
    sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_tiny.pt"
    model_cfg = "./sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
    
    if not os.path.exists(sam2_checkpoint):
        print(f"Hata: Model dosyası bulunamadı: {sam2_checkpoint}")
        return
    
    if not os.path.exists(model_cfg):
        print(f"Hata: Config dosyası bulunamadı: {model_cfg}")
        return
    
    print(f"Model dosyaları bulundu:")
    print(f"  Checkpoint: {sam2_checkpoint}")
    print(f"  Config: {model_cfg}")
    
    # Model yükleme
    print("\nModel yükleniyor...")
    try:
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        print("Model başarıyla yüklendi!")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return
    
    # Dummy resim oluştur (PIL olmadan)
    print("\nDummy resim oluşturuluyor...")
    height, width = 256, 256
    
    # Basit bir gradient resim oluştur
    dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Gradient ekle
    for i in range(height):
        for j in range(width):
            dummy_image[i, j, 0] = int(255 * i / height)  # Kırmızı gradient
            dummy_image[i, j, 1] = int(255 * j / width)   # Yeşil gradient
            dummy_image[i, j, 2] = 128                    # Sabit mavi
    
    print(f"Dummy resim boyutu: {dummy_image.shape}")
    
    # Resmi işleme
    print("\nResim işleniyor...")
    predictor.set_image(dummy_image)
    
    # Örnek 1: Tek nokta ile segmentasyon
    print("\n1. Tek nokta ile segmentasyon...")
    h, w = dummy_image.shape[:2]
    input_point = np.array([[w//2, h//2]])  # Resmin ortası
    input_label = np.array([1])  # Foreground point
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    # En iyi mask'i seç
    best_mask_idx = np.argmax(scores)
    best_mask = masks[best_mask_idx]
    
    print(f"   Nokta koordinatı: {input_point[0]}")
    print(f"   Bulunan mask sayısı: {len(masks)}")
    print(f"   En iyi mask skoru: {scores[best_mask_idx]:.3f}")
    print(f"   Mask boyutu: {best_mask.shape}")
    print(f"   Mask veri tipi: {best_mask.dtype}")
    
    # Örnek 2: Bounding box ile segmentasyon
    print("\n2. Bounding box ile segmentasyon...")
    margin = min(w, h) // 4
    input_box = np.array([margin, margin, w-margin, h-margin])
    
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    
    print(f"   Box koordinatları: [{input_box[0]}, {input_box[1]}, {input_box[2]}, {input_box[3]}]")
    print(f"   Mask skoru: {scores[0]:.3f}")
    print(f"   Mask boyutu: {masks[0].shape}")
    
    # Örnek 3: Çoklu nokta ile segmentasyon
    print("\n3. Çoklu nokta ile segmentasyon...")
    input_points = np.array([
        [w//3, h//3],      # Sol üst
        [2*w//3, 2*h//3],  # Sağ alt
    ])
    input_labels = np.array([1, 1])  # Her ikisi de foreground
    
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )
    
    print(f"   Nokta 1: [{input_points[0, 0]}, {input_points[0, 1]}]")
    print(f"   Nokta 2: [{input_points[1, 0]}, {input_points[1, 1]}]")
    print(f"   Mask skoru: {scores[0]:.3f}")
    print(f"   Mask boyutu: {masks[0].shape}")
    
    # Mask istatistikleri
    print(f"\nMask İstatistikleri (En iyi mask):")
    print(f"   Unique değerler: {np.unique(best_mask)}")
    print(f"   True piksel sayısı: {np.sum(best_mask)}")
    print(f"   False piksel sayısı: {np.sum(~best_mask)}")
    print(f"   Segmentasyon oranı: {np.sum(best_mask) / best_mask.size * 100:.2f}%")
    
    # Mask'i numpy array olarak kaydet
    mask_filename = "sam2_mask_output.npy"
    np.save(mask_filename, best_mask)
    print(f"\nEn iyi mask '{mask_filename}' dosyasına kaydedildi.")
    print(f"Mask'i yüklemek için: mask = np.load('{mask_filename}')")
    
    print(f"\nSAM2 Testi başarıyla tamamlandı!")
    print(f"Model çalışıyor ve segmentasyon yapabiliyor.")

if __name__ == "__main__":
    main()