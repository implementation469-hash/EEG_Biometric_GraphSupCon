import os
import argparse
import numpy as np
import torch
from src.model_graphsupcon import GraphSupConEEGNet

def extract_embeddings(model, X, device, batch_size=256):
    model.eval()
    E = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i:i+batch_size]).to(device)
            emb, _ = model(xb, return_emb=True)
            E.append(emb.cpu().numpy())
    return np.concatenate(E, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz_path", required=True)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--probe_session", type=int, default=1)
    ap.add_argument("--max_samples", type=int, default=8000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y_subj = data["y_subj"].astype(int)
    y_sess = data["y_sess"].astype(int)

    idx = np.where(y_sess == args.probe_session)[0]
    if len(idx) > args.max_samples:
        idx = np.random.choice(idx, args.max_samples, replace=False)

    Xp = X[idx]
    Yp = y_subj[idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphSupConEEGNet().to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    E = extract_embeddings(model, Xp, device)

    out_path = os.path.join(args.out_dir, "stage3_embeddings_probe_session1.npz")
    np.savez_compressed(out_path, E=E, Y=Yp, idx=idx)

    print("Saved:", out_path)
    print("E shape:", E.shape)

if __name__ == "__main__":
    main()
