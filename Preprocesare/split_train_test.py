import json
from sklearn.model_selection import train_test_split  # (Instalează cu: pip install scikit-learn)

# --- CONFIGURARE ---
# Acesta este fișierul mare, filtrat, creat de scriptul anterior
INPUT_JSON = r'C:\Users\Sebi\Desktop\DB_MLAV\Imagini_Filtrate_Verificare\face_dataset_10k.json'

# Numele fișierelor de output
OUTPUT_TRAIN_JSON = 'face_train_75p.json'
OUTPUT_TEST_JSON = 'face_test_25p.json'

TEST_SIZE = 0.25  # 25% pentru testare
RANDOM_STATE = 42  # Pentru rezultate reproductibile
# --- SFÂRȘIT CONFIGURARE ---

print(f"Se încarcă setul de date filtrat: {INPUT_JSON}...")
with open(INPUT_JSON, 'r') as f:
    data = json.load(f)

# 1. Obține toate ID-urile unice ale imaginilor
all_image_ids = [img['id'] for img in data['images']]
print(f"Total imagini de împărțit: {len(all_image_ids)}")
print(f"Total adnotări de împărțit: {len(data['annotations'])}")

# 2. Împarte ID-urile imaginilor
train_ids, test_ids = train_test_split(
    all_image_ids,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# 3. Transformă-le în seturi (set) pentru căutare rapidă O(1)
train_id_set = set(train_ids)
test_id_set = set(test_ids)
print(f"Imagini de antrenare: {len(train_id_set)}")
print(f"Imagini de testare: {len(test_id_set)}")

# 4. Creează noile structuri de date
# Începem cu categoriile, care sunt identice în ambele
base_categories = data.get('categories', [])
train_data = {'images': [], 'annotations': [], 'categories': base_categories}
test_data = {'images': [], 'annotations': [], 'categories': base_categories}

# 5. Adaugă imaginile în seturile corecte
for img in data['images']:
    if img['id'] in train_id_set:
        train_data['images'].append(img)
    elif img['id'] in test_id_set:
        test_data['images'].append(img)

# 6. Adaugă adnotările în seturile corecte
# Iterăm prin toate adnotările și le asignăm pe baza image_id-ului lor
for ann in data['annotations']:
    if ann['image_id'] in train_id_set:
        train_data['annotations'].append(ann)
    elif ann['image_id'] in test_id_set:
        test_data['annotations'].append(ann)

# 7. Salvează fișierele finale
print("\nSalvare fișiere...")

print(f"Salvare train... ({len(train_data['images'])} imagini, {len(train_data['annotations'])} adnotări)")
with open(OUTPUT_TRAIN_JSON, 'w') as f:
    json.dump(train_data, f)

print(f"Salvare test... ({len(test_data['images'])} imagini, {len(test_data['annotations'])} adnotări)")
with open(OUTPUT_TEST_JSON, 'w') as f:
    json.dump(test_data, f)

print("\nÎmpărțire finalizată cu succes!")
print(f"Fișier Train: {OUTPUT_TRAIN_JSON}")
print(f"Fișier Test: {OUTPUT_TEST_JSON}")