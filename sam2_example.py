#!/usr/bin/env python3
"""
SAM2 (Segment Anything Model 2) Örnek Script
Bu script SAM2 modelini kullanarak resim segmentasyonu yapar.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

# SAM2 modüllerini import et
sys.path.append('./sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def show_mask(mask, ax, random_color=False, borders=True):
    """Mask'i görselleştir"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        try:
            import cv2
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
        except ImportError:
            print("OpenCV bulunamadı, border olmadan gösterilecek")
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    """Noktaları görselleştir"""
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
               s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    """Kutuyu görselleştir"""
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                              facecolor=(0, 0, 0, 0), lw=2))

def main():
    """Ana fonksiyon"""
    print("SAM2 Örnek Script")
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
        image = np.array(image.convert("RGB"))
        print(f"Resim boyutu: {image.shape}")
    except Exception as e:
        print(f"Resim yükleme hatası: {e}")
        return
    
    # Orijinal resmi göster
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Orijinal Resim")
    plt.axis('off')
    
    # Resmi işleme
    print("\nResim işleniyor...")
    predictor.set_image(image)
    
    # Örnek 1: Tek nokta ile segmentasyon
    print("\n1. Tek nokta ile segmentasyon...")
    h, w = image.shape[:2]
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
    
    plt.subplot(2, 2, 2)
    plt.imshow(image)
    show_mask(best_mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Tek Nokta Segmentasyon (Score: {scores[best_mask_idx]:.3f})")
    plt.axis('off')
    
    # Örnek 2: Bounding box ile segmentasyon
    print("\n2. Bounding box ile segmentasyon...")
    # Resmin merkezinde bir kutu tanımla
    margin = min(w, h) // 4
    input_box = np.array([margin, margin, w-margin, h-margin])
    
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    
    plt.subplot(2, 2, 3)
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.title(f"Box Segmentasyon (Score: {scores[0]:.3f})")
    plt.axis('off')
    
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
    
    plt.subplot(2, 2, 4)
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_points(input_points, input_labels, plt.gca())
    plt.title(f"Çoklu Nokta Segmentasyon (Score: {scores[0]:.3f})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sam2_example_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nSegmentasyon tamamlandı!")
    print(f"Sonuçlar 'sam2_example_results.png' dosyasına kaydedildi.")
    print(f"En iyi mask boyutu: {best_mask.shape}")
    print(f"Unique pixel değerleri: {np.unique(best_mask)}")

if __name__ == "__main__":
    main()