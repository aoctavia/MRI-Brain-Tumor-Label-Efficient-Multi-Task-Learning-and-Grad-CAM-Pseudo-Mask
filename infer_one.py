import argparse, torch, cv2, numpy as np
from pathlib import Path

# Simple transforms (match training normalization if possible)
def preprocess(img, size=224):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    # Normalize with ImageNet stats (adjust if you used others)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # CHW
    return torch.from_numpy(img).unsqueeze(0)

def overlay_mask(original_bgr, mask, alpha=0.35):
    # mask expected in [0,1], single channel, same size as original
    heat = (mask * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    out = cv2.addWeighted(original_bgr, 1.0, heat, alpha, 0)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained .pt/.pth model (torch.load)")
    ap.add_argument("--image", required=True, help="Path to input image (PNG/JPG)")
    ap.add_argument("--out", required=True, help="Where to save overlay (PNG)")
    ap.add_argument("--classes", nargs="+", default=["glioma","meningioma","no_tumor","pituitary_tumor"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Load
    device = torch.device(args.device)
    model = torch.load(args.model, map_location=device)
    model.eval()

    # Read image
    img_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    x = preprocess(img_bgr).to(device)

    # Forward
    with torch.no_grad():
        out = model(x)
        # Support either dictionary or tuple return
        if isinstance(out, dict):
            logits = out.get("logits", None)
            seg    = out.get("seg", None)
        elif isinstance(out, (list, tuple)) and len(out) >= 2:
            logits, seg = out[0], out[1]
        else:
            logits, seg = out, None

        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_idx = int(np.argmax(probs))
        pred_name = args.classes[pred_idx]

    print("Prediction:")
    for i, c in enumerate(args.classes):
        print(f"- {c:16s}: {probs[i]:.4f}")
    print(f"Top-1: {pred_name}")

    # If segmentation head exists, save overlay
    if seg is not None:
        seg = torch.sigmoid(seg).squeeze().detach().cpu().numpy()
        # Normalize to [0,1]
        seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-6)
        seg_resized = cv2.resize(seg, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        overlay = overlay_mask(img_bgr, seg_resized, alpha=0.35)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, overlay)
        print(f"Overlay saved to: {args.out}")
    else:
        print("Segmentation head not found; no overlay saved.")

if __name__ == "__main__":
    main()
