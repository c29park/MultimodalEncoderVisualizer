import numpy as np
import matplotlib.pyplot as plt
from detect_individuals import build_face_db, embed_face
from PIL import Image
# face_db: {person: np.ndarray (N_images,512)} of normalized CLIP embeddings
# Build lists of all intra- and inter-person similarities
def main():    
    face_db, path_db = build_face_db("/home/user/chp/videoDataset/person_faces")
    print("Loaded identities:", list(face_db.keys()))
    print("Examples per person:", {p: len(face_db[p]) for p in face_db})
    intra, inter = [], []
    persons = list(face_db.keys())

    for person in persons:
        refs = face_db[person]
        # all pairwise sims within this person
        for i in range(len(refs)):
            for j in range(i+1, len(refs)):
                intra.append(refs[i].dot(refs[j]))

    for i in range(len(persons)):
        for j in range(i+1, len(persons)):
            a = face_db[persons[i]]
            b = face_db[persons[j]]
            # sample a few cross-pairs
            for v in a[:3]:
                for w in b[:3]:
                    inter.append(v.dot(w))

    # plot histograms
    plt.hist(intra, bins=30, alpha=0.6, label="same-person")
    plt.hist(inter, bins=30, alpha=0.6, label="diff-person")
    plt.legend()
    plt.xlabel("Cosine similarity")
    plt.show()

if __name__ == "__main__":
    main()