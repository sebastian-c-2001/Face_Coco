import json
import os
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm # Pentru o bară de progres

INPUT_JSON = r'C:\Users\Sebi\Desktop\DB_MLAV\train2017\coco_wholebody_train_v1.0.json'
IMAGE_DIR = r'C:\Users\Sebi\Desktop\DB_MLAV\train2017'

# Indecșii pentru ochi (modelul de 68 puncte)
EYE_INDICES = list(range(36, 48))

print("Se încarcă JSON-ul original (poate dura)...")
with open(INPUT_JSON, 'r') as f:
    data = json.load(f)

print("Se creează indexul de imagini...")
image_map = {img['id']: img for img in data['images']}

# --- AICI ESTE MAGIA ---
print("Se grupează adnotările după ID-ul imaginii...")
# Vom crea un dicționar: {image_id_1: [ann_1, ann_2], image_id_2: [ann_3], ...}
image_id_to_annotations = defaultdict(list)

# Folosim tqdm pentru a vedea progresul
for ann in tqdm(data['annotations'], desc="Grupare adnotări"):
    image_id_to_annotations[ann['image_id']].append(ann)

print(f"S-au găsit adnotări pentru {len(image_id_to_annotations)} imagini.")
# --- SFÂRȘIT PRE-PROCESARE ---


print("Se caută prima imagine CU O SINGURĂ PERSOANĂ și OCHI VIZIBILI (v=2)...")
found = False

# Iterăm prin dicționarul nostru grupat
for image_id, annotation_list in image_id_to_annotations.items():

    # --- 1. FILTRUL "O SINGURĂ PERSOANĂ" ---
    if len(annotation_list) != 1:
        continue  # Sărim peste; are 0 sau mai multe persoane

    # Dacă am ajuns aici, știm că 'annotation_list' are fix un element
    ann = annotation_list[0]

    # --- 2. FILTRUL "OCHI VIZIBILI" (la fel ca înainte) ---
    if not ann.get('face_valid', False):
        continue
    if 'face_kpts' not in ann or ann['face_kpts'] is None:
        continue

    all_face_kpts = ann['face_kpts']
    has_visible_eyes = False

    for idx in EYE_INDICES:
        v_idx = idx * 3 + 2
        if v_idx >= len(all_face_kpts):
            has_visible_eyes = False
            break
        if all_face_kpts[v_idx] == 2:
            has_visible_eyes = True
            break

            # --- 3. ACȚIUNEA (Dacă am găsit!) ---
    if has_visible_eyes:
        print("\n" + "=" * 50)
        print(" S-A GĂSIT O IMAGINE: 1 PERSOANĂ / OCHI VIZIBILI (v=2) ")
        print("=" * 50)

        image_info = image_map[image_id]

        print("\n--- INFORMAȚII IMAGINE (din 'images') ---")
        print(json.dumps(image_info, indent=4))

        print("\n--- INFORMAȚII ADNOTARE (din 'annotations') ---")
        print(json.dumps(ann, indent=4))
        print("=" * 50)

        # Afișare vizuală
        file_name = image_info['file_name']
        image_path = os.path.join(IMAGE_DIR, file_name)

        print(f"\nSe încarcă imaginea: {image_path}")

        my_eye_keypoints_coords = []
        print("\nCoordonate Puncte Ochi (x, y, vizibilitate):")
        for idx in EYE_INDICES:
            x = all_face_kpts[idx * 3]
            y = all_face_kpts[idx * 3 + 1]
            v = all_face_kpts[idx * 3 + 2]
            my_eye_keypoints_coords.append((x, y, v))
            print(f"  Punct {idx}: ({x:.1f}, {y:.1f}, {v})")

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for (x, y, v) in my_eye_keypoints_coords:
            if v == 2:  # Verde
                cv2.circle(image_rgb, (int(x), int(y)), radius=4, color=(0, 255, 0), thickness=-1)
            elif v == 1:  # Roșu
                cv2.circle(image_rgb, (int(x), int(y)), radius=4, color=(255, 0, 0), thickness=-1)

        print("\nSe afișează imaginea... (închide fereastra pentru a continua)")
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.title(f"Imagine cu 1 Persoană: {file_name}")
        plt.axis('off')
        plt.show()

        found = True
        break  # Oprim bucla principală

if not found:
    print("NU s-a găsit nicio imagine care să conțină DOAR O PERSOANĂ cu OCHI VIZIBILI (v=2).")
    print("NU s-a găsit nicio adnotare cu OCHI VIZIBILI (v=2) în tot setul de date.")
    print("Verifică dacă fișierul JSON și căile sunt corecte.")