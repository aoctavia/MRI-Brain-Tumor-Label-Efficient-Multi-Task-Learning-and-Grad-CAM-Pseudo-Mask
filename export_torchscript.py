import argparse, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to trained .pt/.pth (torch.load)")
    ap.add_argument("--out", required=True, help="Path to save scripted model (.pt)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = torch.load(args.model, map_location=device)
    model.eval()

    # Dummy input; adjust size/channels if model expects different
    dummy = torch.randn(1, 3, 224, 224, device=device)

    try:
        scripted = torch.jit.trace(model, dummy, strict=False)
    except Exception:
        scripted = torch.jit.script(model)

    scripted.save(args.out)
    print(f"Saved TorchScript: {args.out}")

if __name__ == "__main__":
    main()
