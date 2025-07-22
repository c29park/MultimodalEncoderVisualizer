import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from skimage import io

FACE_DB_FOLDER = "/home/user/chp/videoDataset/person_faces"
DEVICE = "cpu"


fa = FaceAnalysis(
    allowed_modules = ['detection', 'recognition'],
    name='antelope'
)
fa.prepare(ctx_id=0, det_size=(640,640))

def embed_face(img_path: str) -> tuple[np.ndarray, str] | None:
    """Reads an img from disk, detects face, and returns (normalized 512-d embedding, img_path)"""
    img = io.imread(img_path)
    if not faces:
        return None
    
    face = max(faces, key = lambda x: x.bbox[2] * x.bbox[3])
    emb = face.normed_embedding
    return emb.astype('float32'), img_path

def build_face_db(db_folder: str=FACE_DB_FOLDER) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Scan each person-folder, compute embeddings for all images, and store as a (N_images x 512) array"""
    face_db = {}
    path_db = {}
    for person in os.listdir(db_folder):
        person_dir = os.path.join(db_folder, person)
        
        if not os.path.isdir(person_dir):
            continue
        embs, paths= [], []
        for fname in os.listdir(person_dir):
            path = os.path.join(person_dir, fname)
            try:
                out = embed_face(path)
                if out is None:
                    continue
                emb, _ = out
                embs.append(emb)
                paths.append(path)
            except Exception:
                continue
        if embs:
            face_db[person] = np.stack(embs, axis =0)
            path_db[person] = paths
    return face_db, path_db

def identify_face(emb: np.ndarray, face_db: dict[str, np.ndarray], path_db: dict[str, list[str]], k:int =3, threshold_accept: float = 0.6, threshold_reject: float=0.3, epsilon: float = 0.05) -> tuple[str|None, str]:
    """
    Hybrid Face ID:
        - If best_dist < threshold_accept: accept best_name.
        - Elif best_dist > threshold_reject: return None
        - Else: bolderline -> if (second_dist - best_dist) < e, return "ambiguous: A vs B" 
    """
    sims = []
    for person, refs in face_db.items():
        person_sims = refs.dot(emb)
        idx = int(np.argmax(person_sims))
        sims.append((person, float(person_sims[idx])), path_db[person][idx])
    
    sims.sort(key=lambda x: x[1], reverse=True)
    top_k = sims[:k]
    best_name, best_sim, best_path = top_k[0]

    if best_sim > threshold_accept:
        label = best_name
    elif best_sim < threshold_reject:
        label = None
    else:
        if len(top_k) >1 and (best_sim - top_k[1][1]) < epsilon:
            label = f"ambiguous: {best_name} vs {top_k[1][0]}"
        else:
            label = best_name
    
    return label, best_path
        


def detect_people_in_video(
        video_path: str,
        face_db: dict[str, np.ndarray],
        path_db: dict[str, list[str]],
        sample_rate_sec: float = 1.0
) -> list[tuple[str, str]]:
    """
    Scan the video every 'sample_rate_sec' seconds, run hybrid face ID,
    and collect the first (name, best_image_path) for each known person
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * sample_rate_sec))

    seen: dict[str, dict[str, float]] = {}
    for frame_idx in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        #InsightFace 
        img =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        faces = fa.get(rgb)
        if not faces:
            continue
        
        face = max(faces, key = lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        emb = face.normed_embedding

        label, best_img = identify_face(
            emb, face_db, path_db
        )

        if label and label not in seen:
            ts = frame_idx / fps
            seen[label] = {"path": best_img, "timestamp":ts}
            if len(seen) == len(face_db):
                break

    cap.release()

    return [
        (name, info["path"], info["timestamp"])
        for name, info in seen.items()
    ]