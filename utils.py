from PIL import Image
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np

def show_image_and_mask_tensor(image, mask, cmap_mask="gray", alpha=0.5):
    """
    Visualizza immagine e maschera (sovrapposta).
    image: Tensor (C,H,W) o (1,C,H,W)
    mask:  Tensor (H,W) o (1,H,W)
    """

    # Rimuove dimensione batch se presente
    if image.dim() == 4:
        image = image.squeeze(0)
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    # Porta immagine su CPU e in numpy
    image = image.detach().cpu()
    mask = mask.detach().cpu()

    # Se CHW -> HWC
    if image.shape[0] in [1, 3]:
        image = image.permute(1, 2, 0)

    image = image.numpy()
    mask = mask.numpy()

    plt.figure(figsize=(12, 5))

    # Immagine
    plt.subplot(1, 3, 1)
    plt.title("Immagine")
    plt.imshow(image)
    plt.axis("off")

    # Maschera
    plt.subplot(1, 3, 2)
    plt.title("Maschera")
    plt.imshow(mask, cmap=cmap_mask)
    plt.axis("off")

    # Sovrapposizione
    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(image)
    plt.imshow(mask, cmap="jet", alpha=alpha)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def generate_mask_tensor_from_coco_file(coco_data, target_image_id, size=(700, 700)):
    """
    Genera una maschera binaria da COCO per un'immagine specifica.
    Ritorna un tensore Long [H,W] pronto per il training.
    """
    img_info = next((img for img in coco_data['images'] if img['id'] == target_image_id), None)
    if img_info is None:
        raise ValueError(f"ID immagine {target_image_id} non trovato nel JSON.")

    height, width = img_info['height'], img_info['width']
    mask = np.zeros((height, width), dtype=np.uint8)

    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == target_image_id]

    for ann in annotations:
        for seg_points in ann['segmentation']:
            poly = np.array(seg_points).reshape((-1,2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], color=1)  # 1 = cavo, 0 = sfondo

    # Converti in PIL per ridimensionare senza interpolazione
    mask = Image.fromarray(mask)
    mask = mask.resize(size, resample=Image.NEAREST)

    return torch.from_numpy(np.array(mask)).long()  # shape [H,W], dtype Long