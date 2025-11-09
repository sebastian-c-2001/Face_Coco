import json
import os
from tqdm import tqdm  # Pentru o bară de progres drăguță (instalează cu: pip install tqdm)

# --- CONFIGURARE ---
INPUT_JSON = 'path/to/coco_wholebody_train_v1.0.json'
OUTPUT_JSON = 'face_dataset_10k.json'
IMAGE_LIMIT = 10000

# Alege 10-15 puncte. Acesta este exemplul meu.
# TU trebuie să alegi indecșii tăi (între 29 și 96).
YOUR_CHOSEN_KP_INDICES = [
    59,  # Vârf nas
    65,  # Colț ochi stâng
    74,  # Colț ochi drept
    77,  # Colț gură stânga
    83,  # Colț gură dreapta
    89,  # Punct bărbie
    62,  # Pleoapă sus stânga
    71,  # Pleoapă sus dreapta
    92,  # Mijloc buză jos
    68,  # Pleoapă jos stânga
    78  # Pleoapă jos dreapta
]
NUM_MY_KEYPOINTS = len(YOUR_CHOSEN_KP_INDICES)
# --- SFÂRȘIT CONFIGURARE ---

print("Se încarcă JSON-ul original (poate dura)...")
with open(INPUT_JSON, 'r') as f:
    data = json.load(f)

new_annotations = []
new_images = []
added_image_ids = set()  # Folosim un set pentru a găsi rapid ID-urile

print(f"Se procesează {len(data['annotations'])} adnotări...")

for ann in tqdm(data['annotations']):
    # 1. Verificăm dacă am atins limita de imagini
    if len(added_image_ids) >= IMAGE_LIMIT:
        print(f"Limita de {IMAGE_LIMIT} imagini atinsă.")
        break

    # 2. Verificăm dacă adnotarea are puncte 'wholebody'
    if 'wholebody_keypoints' not in ann or ann['wholebody_keypoints'] is None:
        continue

    all_kpts = ann['wholebody_keypoints']  # Lista de 399 numere
    my_keypoints = []
    has_visible_face_points = False

    # 3. Extragem DOAR punctele noastre de interes
    for idx in YOUR_CHOSEN_KP_INDICES:
        x_idx = idx * 3
        y_idx = idx * 3 + 1
        v_idx = idx * 3 + 2

        x = all_kpts[x_idx]
        y = all_kpts[y_idx]
        v = all_kpts[v_idx]

        my_keypoints.extend([x, y, v])

        # Considerăm o adnotare validă dacă are MĂCAR UN punct de față vizibil (v=1 sau v=2)
        if v > 0:
            has_visible_face_points = True

    # 4. Dacă adnotarea este validă, o adăugăm
    if has_visible_face_points:
        # Creăm o adnotare nouă, curată
        new_ann = {
            'id': ann['id'],
            'image_id': ann['image_id'],
            'category_id': ann['category_id'],
            'bbox': ann['bbox'],  # Păstrăm bbox-ul persoanei
            'area': ann['area'],
            'iscrowd': ann['iscrowd'],
            'keypoints': my_keypoints,  # Aici punem lista noastră scurtă de puncte
            'num_keypoints': NUM_MY_KEYPOINTS  # Numărul de puncte (ex: 11)
        }
        new_annotations.append(new_ann)
        added_image_ids.add(ann['image_id'])

print(f"S-au găsit {len(new_annotations)} adnotări valide în {len(added_image_ids)} imagini.")

# 5. Acum adăugăm informațiile despre imaginile filtrate
print("Se filtrează lista de imagini...")
for img in tqdm(data['images']):
    if img['id'] in added_image_ids:
        new_images.append(img)

# 6. (Opțional dar recomandat) Actualizăm categoriile
# Vom crea nume generice pentru punctele tale
keypoint_names = [f"face_pt_{i}" for i in range(NUM_MY_KEYPOINTS)]

new_categories = data['categories']
# Presupunem că e doar o categorie ('person')
if new_categories and len(new_categories) > 0:
    new_categories[0]['keypoints'] = keypoint_names
    new_categories[0]['skeleton'] = []  # Nu avem schelet pentru punctele astea

# 7. Salvăm noul JSON
new_dataset = {
    'images': new_images,
    'annotations': new_annotations,
    'categories': new_categories
}

print(f"Se salvează noul JSON în {OUTPUT_JSON}...")
with open(OUTPUT_JSON, 'w') as f:
    json.dump(new_dataset, f)

print("Procesare finalizată!")