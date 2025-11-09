import json
import os
import shutil
from tqdm import tqdm

# --- CONFIGURE ---
INPUT_JSON = r'C:\Users\Sebi\Desktop\DB_MLAV\train2017\coco_wholebody_train_v1.0.json'
IMAGE_DIR = r'C:\Users\Sebi\Desktop\DB_MLAV\train2017'  # !! VERIFICĂ DACĂ E CORECTĂ !!
OUTPUT_IMAGE_DIR = r'C:\Users\Sebi\Desktop\DB_MLAV\Imagini_Filtrate_Verificare'
OUTPUT_JSON = r'C:\Users\Sebi\Desktop\DB_MLAV\face_dataset_10k.json'
IMAGE_LIMIT = 10000


# Câte puncte faciale (din 68) trebuie să fie etichetate (v>0) minim?
MIN_LABELED_POINTS = 5
# Care e împrăștierea minimă (în pixeli) a acelor puncte ca să nu fie "stivuite"?
MIN_FACE_SPREAD = 10
# ---

# Indecșii ochilor pe care îi vom extrage
EYE_INDICES = list(range(36, 48))
NUM_MY_KEYPOINTS = len(EYE_INDICES)


os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

print("Se încarcă JSON-ul original (poate dura)...")
with open(INPUT_JSON, 'r') as f:
    data = json.load(f)

print("Pasul 1/5: Se creează indexul de imagini...")
image_map = {img['id']: img['file_name'] for img in data['images']}

print(f"Pasul 2/5: Se filtrează adnotările (Limită imagini: {IMAGE_LIMIT})...")
new_annotations = []
added_image_ids = set()

for ann in tqdm(data['annotations'], desc="Filtrare adnotări"):

    if len(added_image_ids) >= IMAGE_LIMIT:
        print(f"\nLimita de {IMAGE_LIMIT} imagini unice atinsă.")
        break

    # Filtru 1: Verificări de bază
    if not ann.get('face_valid', False):
        continue
    if 'face_kpts' not in ann or ann['face_kpts'] is None or len(ann['face_kpts']) == 0:
        continue

    all_face_kpts = ann['face_kpts']

    # --- !!! AICI E NOUL FILTRU "DEȘTEPT" !!! ---
    # Colectăm toate punctele faciale etichetate (v > 0)
    labeled_kpts_coords = []
    for idx in range(68):  # Iterăm prin TOATE 68 de puncte faciale
        v_idx = idx * 3 + 2
        if v_idx < len(all_face_kpts) and all_face_kpts[v_idx] > 0:
            x = all_face_kpts[idx * 3]
            y = all_face_kpts[idx * 3 + 1]
            labeled_kpts_coords.append((x, y))

    # Filtru 2: Verificăm dacă avem suficiente puncte etichetate
    if len(labeled_kpts_coords) < MIN_LABELED_POINTS:
        continue  # Prea puține puncte, probabil invalid

    # Filtru 3: Verificăm împrăștierea (spread) pentru a evita punctele "stivuite"
    xs = [kp[0] for kp in labeled_kpts_coords]
    ys = [kp[1] for kp in labeled_kpts_coords]

    spread_w = max(xs) - min(xs)
    spread_h = max(ys) - min(ys)

    if spread_w < MIN_FACE_SPREAD or spread_h < MIN_FACE_SPREAD:
        continue  # Punctele sunt "stivuite" (ca la autobuz), respingem
    # --- SFÂRȘIT FILTRU "DEȘTEPT" ---

    # --- DACĂ AM TRECUT DE FILTRE, PROCESĂM ADNOTAREA ---

    # Extragem DOAR cele 12 puncte ale OCHILOR
    my_keypoints = []
    for idx in EYE_INDICES:
        x = all_face_kpts[idx * 3]
        y = all_face_kpts[idx * 3 + 1]
        v = all_face_kpts[idx * 3 + 2]
        my_keypoints.extend([x, y, v])

    face_box = ann.get('face_box', ann['bbox'])

    new_ann = {
        'id': ann['id'],
        'image_id': ann['image_id'],
        'category_id': ann['category_id'],
        'bbox': face_box,
        'area': face_box[2] * face_box[3],
        'iscrowd': ann['iscrowd'],
        'keypoints': my_keypoints,
        'num_keypoints': NUM_MY_KEYPOINTS
    }
    new_annotations.append(new_ann)

    # Logica de copiere a imaginilor
    is_new_image = ann['image_id'] not in added_image_ids
    added_image_ids.add(ann['image_id'])

    if is_new_image:
        try:
            file_name = image_map[ann['image_id']]
            src_path = os.path.join(IMAGE_DIR, file_name)
            dst_path = os.path.join(OUTPUT_IMAGE_DIR, file_name)
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
        except (KeyError, FileNotFoundError):
            pass  # Ignorăm erorile de copiere în tăcere

print(f"\nS-au găsit {len(new_annotations)} adnotări valide în {len(added_image_ids)} imagini unice.")

# --- PASUL 3: Filtrarea listei de imagini ---
print("Pasul 3/5: Se filtrează lista de imagini...")
new_images = []
for img in tqdm(data['images'], desc="Filtrare listă imagini"):
    if img['id'] in added_image_ids:
        new_images.append(img)

# --- PASUL 4: Crearea categoriilor ---
print("Pasul 4/5: Se actualizează categoriile...")
keypoint_names = [f"eye_pt_{i}" for i in range(NUM_MY_KEYPOINTS)]
new_categories = [{
    'id': 1,
    'name': 'person',
    'supercategory': 'person',
    'keypoints': keypoint_names,
    'skeleton': []
}]

# --- PASUL 5: Salvarea noului JSON ---
new_dataset = {
    'images': new_images,
    'annotations': new_annotations,
    'categories': new_categories
}

print(f"Pasul 5/5: Se salvează noul JSON în {OUTPUT_JSON}...")
with open(OUTPUT_JSON, 'w') as f:
    json.dump(new_dataset, f)

print("Procesare finalizată cu succes!")
print(f"Noul tău set de date este gata la: {OUTPUT_JSON}")
print(f"Imaginile pentru verificare au fost copiate în: {OUTPUT_IMAGE_DIR}")