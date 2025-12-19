# src/embed_extract.py

import argparse
import numpy as np
import torch
from model_graphsupcon import GraphSupConEEGNet

def extract_embeddings(model, X, batch_size=256):
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i+batch_size]).float().cuda()
            e = model(xb, return_emb=True)
            embs.append(e.cpu().numpy())
    return np.vstack(embs)

def main(args):
    data = np.load(args.npz_path)
    X = data["X"]

    model = GraphSupConEEGNet(
        num_channels=X.shape[1],
        num_classes=109,
        emb_dim=128
    ).cuda()

    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["model"])
    print("✅ Loaded checkpoint")

    E = extract_embeddings(model, X)
    np.save(args.out_dir + "/embeddings.npy", E)
    print("✅ Saved embeddings:", E.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args)
