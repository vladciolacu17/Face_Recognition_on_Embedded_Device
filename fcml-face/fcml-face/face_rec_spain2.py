from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms

# ==============================
# CONFIG
# ==============================
BASE = Path("/home/vlad/Desktop/fcml-face")
DATASET_ROOT = BASE / "dataset"
COUNTRY = "Spain"
CACHE_DIR = BASE / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = BASE / "results_spain.csv"


# ==============================
# UTILS
# ==============================
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def topk_by_cosine(query_emb: np.ndarray, db_embs: np.ndarray, k: int = 3):
    q = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    db = db_embs / (np.linalg.norm(db_embs, axis=1, keepdims=True) + 1e-9)
    sims = db @ q
    idx = np.argsort(-sims)[:k]
    return idx.tolist()


def person_from_path(p: Path, country_root: Path) -> str:
    # dataset/Spain/<Person>/ref|test/img.jpg  -> <Person>
    return p.parents[1].name


# ==============================
# FACE EMBEDDER
# ==============================
class FaceEmbedder:
    def __init__(self, device="cpu", image_size=160, use_mtcnn=True):
        self.device = torch.device(device)
        self.use_mtcnn = use_mtcnn
        self.image_size = image_size

        if use_mtcnn:
            self.mtcnn = MTCNN(
                image_size=image_size,
                margin=0,
                post_process=True,
                select_largest=True,
                keep_all=False,
                device=self.device,
            )
        else:
            self.mtcnn = None  # disabled

        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    @torch.no_grad()
    def embed_image(self, img: Image.Image):
        if self.use_mtcnn:
            face = self.mtcnn(img)
            if face is None:
                return None
        else:
            # skip face detection, just resize + normalize manually
            img = img.resize((self.image_size, self.image_size))
            tform = transforms.ToTensor()
            face = tform(img).unsqueeze(0).to(self.device)

        emb = self.model(face).cpu().numpy().squeeze(0)
        return emb

# ==============================
# BUILD REFERENCE INDEX (+ CACHE)
# ==============================
def build_reference_index(dataset_root: Path, country: str, cache_dir: Path):
    cache_file = cache_dir / f"{country}_ref_index.npz"
    if cache_file.exists():
        data = np.load(cache_file, allow_pickle=True)
        return data["embeddings"], data["paths"].tolist(), data["labels"].tolist()

    country_root = dataset_root / country
    embedder = FaceEmbedder(device="cpu")

    ref_imgs = []
    for person_dir in country_root.iterdir():
        if not person_dir.is_dir():
            continue
        ref_dir = person_dir / "ref"
        if ref_dir.exists():
            ref_imgs += sorted(p for p in ref_dir.iterdir() if p.is_file())

    print(f"[REF] Found {len(ref_imgs)} reference images. Embedding...")
    embeddings, labels, paths = [], [], []
    for p in ref_imgs:
        img = Image.open(p).convert("RGB")
        emb = embedder.embed_image(img)
        if emb is None:
            print(f"[WARN] No face in reference: {p}")
            continue
        embeddings.append(emb)
        labels.append(person_from_path(p, country_root))
        paths.append(p)

    embeddings = np.stack(embeddings, axis=0)
    np.savez_compressed(
        cache_file,
        embeddings=embeddings,
        paths=np.array(paths, dtype=object),
        labels=np.array(labels, dtype=object),
    )
    print(f"[REF] Cached to {cache_file}")
    return embeddings, paths, labels


# ==============================
# EVALUATE TEST IMAGES
# ==============================
def evaluate_tests(dataset_root: Path, country: str,
                   ref_embs: np.ndarray, ref_paths, ref_labels, out_csv: Path):
    country_root = dataset_root / country
    embedder = FaceEmbedder(device="cpu")

    test_imgs = []
    for person_dir in country_root.iterdir():
        if not person_dir.is_dir():
            continue
        test_dir = person_dir / "test"
        if test_dir.exists():
            test_imgs += sorted(p for p in test_dir.iterdir() if p.is_file())

    rows = []
    print(f"[TEST] Found {len(test_imgs)} test images. Matching...")
    for p in test_imgs:
        img = Image.open(p).convert("RGB")
        emb = embedder.embed_image(img)
        gt = person_from_path(p, country_root)

        if emb is None:
            print(f"[WARN] No face in test: {p}")
            rows.append({
                "test_image": str(p),
                "test_person_groundtruth": gt,
                "match1_label": "", "match1_path": "", "match1_similarity": "",
                "match2_label": "", "match2_path": "", "match2_similarity": "",
                "match3_label": "", "match3_path": "", "match3_similarity": "",
            })
            continue

        top_idx = topk_by_cosine(emb, ref_embs, k=3)
        sims = [cosine_sim(emb, ref_embs[i]) for i in top_idx]

        rows.append({
            "test_image": str(p),
            "test_person_groundtruth": gt,
            "match1_label": ref_labels[top_idx[0]],
            "match1_path": str(ref_paths[top_idx[0]]),
            "match1_similarity": sims[0],
            "match2_label": ref_labels[top_idx[1]],
            "match2_path": str(ref_paths[top_idx[1]]),
            "match2_similarity": sims[1],
            "match3_label": ref_labels[top_idx[2]],
            "match3_path": str(ref_paths[top_idx[2]]),
            "match3_similarity": sims[2],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[OUT] Written to {out_csv}")
    return df


# ==============================
# MAIN
# ==============================
def main():
    # build/load reference index
    ref_embs, ref_paths, ref_labels = build_reference_index(
        DATASET_ROOT, COUNTRY, CACHE_DIR
    )

    # evaluate tests
    df = evaluate_tests(
        DATASET_ROOT, COUNTRY, ref_embs, ref_paths, ref_labels, OUT_CSV
    )

    # pretty console summary WITH FILENAMES
    for _, r in df.iterrows():
        test_name = Path(r["test_image"]).name
        print(f"\nTest: {test_name}")
        print(f" GT: {r['test_person_groundtruth']}")
        print(
            f" 1: {r['match1_label']}  ({Path(r['match1_path']).name})  "
            f"similarity = {float(r['match1_similarity']):.3f}"
        )
        print(
            f" 2: {r['match2_label']}  ({Path(r['match2_path']).name})  "
            f"similarity = {float(r['match2_similarity']):.3f}"
        )
        print(
            f" 3: {r['match3_label']}  ({Path(r['match3_path']).name})  "
            f"similarity = {float(r['match3_similarity']):.3f}"
        )


if __name__ == "__main__":
    main()
