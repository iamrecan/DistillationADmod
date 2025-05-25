#!/usr/bin/env python3
"""
SAM2 (Segment Anything Model 2) Basit Örnek Script
Bu script SAM2 modelini kullanarak resim segmentasyonu yapar.
Sadece temel kütüphaneler kullanır.
"""

import os
import sys
import numpy as np
from PIL import Image
import torch

# SAM2 modüllerini import et
sys.path.append('./sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def save_mask_image(mask, original_image, output_path):
    """Mask'i resim olarak kaydet"""
    # Mask'i normalize et
    mask_normalized = (mask * 255).astype(np.uint8)
    
    # RGB formatına çevir
    mask_rgb = np.stack([mask_normalized, mask_normalized, mask_normalized], axis=-1)
    
    # Orijinal resimle karıştır
    if original_image.shape[:2] == mask.shape:
        alpha = 0.6
        blended = (alpha * original_image + (1-alpha) * mask_rgb).astype(np.uint8)
        
        # PIL Image olarak kaydet
        result_image = Image.fromarray(blended)
        result_image.save(output_path)
        print(f"Sonuç kaydedildi: {output_path}")
    else:
        print(f"Boyut uyuşmazlığı: resim {original_image.shape}, mask {mask.shape}")

def main():
    """Ana fonksiyon"""
    print("SAM2 Basit Örnek Script")
    print("=" * 50)
    
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
    
    # Model yükleme
    print("\nModel yükleniyor...")
    sam2_checkpoint = "./sam2/checkpoints/sam2.1_hiera_tiny.pt"  # En küçük model
    model_cfg = "./sam2/configs/sam2.1/sam2.1_hiera_t.yaml"
    
    try:
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        print("Model başarıyla yüklendi!")
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return
    
    # Resim yükleme
    print("\nResim yükleniyor...")
    image_path = "dummy_image.png"
    
    try:
        image = Image.open(image_path)
        image_np = np.array(image.convert("RGB"))
        print(f"Resim boyutu: {image_np.shape}")
    except Exception as e:
        print(f"Resim yükleme hatası: {e}")
        return
    
    # Resmi işleme
    print("\nResim işleniyor...")
    predictor.set_image(image_np)
    
    # Örnek 1: Tek nokta ile segmentasyon
    print("\n1. Tek nokta ile segmentasyon...")
    h, w = image_np.shape[:2]
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
    
    print(f"   Bulunan mask sayısı: {len(masks)}")
    print(f"   En iyi mask skoru: {scores[best_mask_idx]:.3f}")
    print(f"   Mask boyutu: {best_mask.shape}")
    
    # Sonucu kaydet
    save_mask_image(best_mask, image_np, "sam2_result_point.png")
    
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
    
    print(f"   Box koordinatları: {input_box}")
    print(f"   Mask skoru: {scores[0]:.3f}")
    print(f"   Mask boyutu: {masks[0].shape}")
    
    # Sonucu kaydet
    save_mask_image(masks[0], image_np, "sam2_result_box.png")
    
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
    
    print(f"   Nokta koordinatları: {input_points}")
    print(f"   Mask skoru: {scores[0]:.3f}")
    print(f"   Mask boyutu: {masks[0].shape}")
    
    # Sonucu kaydet
    save_mask_image(masks[0], image_np, "sam2_result_multipoint.png")
    
    # Mask istatistikleri
    print(f"\nMask İstatistikleri:")
    print(f"   Unique değerler: {np.unique(best_mask)}")
    print(f"   True piksel sayısı: {np.sum(best_mask)}")
    print(f"   False piksel sayısı: {np.sum(~best_mask)}")
    print(f"   Segmentasyon oranı: {np.sum(best_mask) / best_mask.size * 100:.2f}%")
    
    print(f"\nSegmentasyon tamamlandı!")
    print(f"Sonuçlar aşağıdaki dosyalara kaydedildi:")
    print(f"   - sam2_result_point.png (tek nokta)")
    print(f"   - sam2_result_box.png (bounding box)")
    print(f"   - sam2_result_multipoint.png (çoklu nokta)")

if __name__ == "__main__":
    main()