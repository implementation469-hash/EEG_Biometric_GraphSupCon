import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data = np.load(args.emb_npz)
    E = data["E"]
    Y = data["Y"]

    tsne = TSNE(n_components=2, init="pca", learning_rate="auto")
    Z = tsne.fit_transform(E)

    plt.scatter(Z[:,0], Z[:,1], s=5)
    plt.title("Stage-3 t-SNE embeddings")
    plt.savefig(os.path.join(args.out_dir, "tsne_embeddings.png"), dpi=200)
    plt.close()

    sil = silhouette_score(E, Y)
    db = davies_bouldin_score(E, Y)

    df = pd.DataFrame([{
        "samples": len(E),
        "subjects": len(np.unique(Y)),
        "silhouette": sil,
        "davies_bouldin": db
    }])

    df.to_csv(os.path.join(args.out_dir, "stage3_metrics.csv"), index=False)
    print(df)

if __name__ == "__main__":
    main()
