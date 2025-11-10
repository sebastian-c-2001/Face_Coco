import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2  # Pentru detecție facială și desenare
import matplotlib.pyplot as plt
import os

# --- 1. CONFIGURARE ---
MODEL_PATH = r'C:\PycharmProjects\Face_Coco\train\best_model_face_keypoints.pth'
IMAGE_TO_TEST_PATH = r'C:\Users\Sebi\Desktop\DB_MLAV\test4.jpg'  # <-- SCHIMBĂ ASTA
IMAGE_SIZE = 224

# Calea către clasificatorul Haar Cascade (vine cu OpenCV)
# Acesta este necesar pentru a GĂSI fața în imagine
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


# --- 2. RE-CREAREA ARHITECTURII MODELULUI ---
# (Această funcție trebuie să fie identică cu cea din scriptul de antrenament)
def get_model():
    model = resnet18(weights=None)  # Nu mai avem nevoie de 'weights=DEFAULT'
    num_features = model.fc.in_features
    num_outputs = 12 * 2  # 12 puncte (x, y)
    model.fc = nn.Linear(num_features, num_outputs)
    return model


# --- 3. DEFINIREA TRANSFORMĂRILOR ---
# (Trebuie să fie IDENTICE cu cele de la validare/antrenament)
data_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


# --- 4. FUNCȚIA PRINCIPALĂ DE PREDICTIE ---
def predict_keypoints(image_path, model, device):
    # --- A. Detectează Fața ---
    # Încărcăm imaginea originală cu OpenCV pentru detecție
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"EROARE: Nu am putut încărca imaginea de la: {image_path}")
        return

    # Convertim în grayscale pentru detectorul Haar
    image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Încărcăm detectorul de fețe
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    # Detectăm fețele
    # 'faces' este o listă de [x, y, w, h]
    faces = face_cascade.detectMultiScale(image_gray, 1.1, 4)

    if len(faces) == 0:
        print("Nu a fost detectată nicio față în imagine.")
        return

    # Încărcăm imaginea originală cu PIL pentru procesare (așa am făcut la train)
    original_pil_image = Image.open(image_path).convert('RGB')

    # Vom crea o copie a imaginii CV pentru a desena pe ea
    image_with_kpts = cv2.cvtColor(np.array(original_pil_image), cv2.COLOR_RGB2BGR)

    print(f"S-au detectat {len(faces)} fețe. Se procesează...")
    print("=" * 30)

    # Iterăm prin fiecare față detectată
    for i, (x, y, w, h) in enumerate(faces):
        print(f"--- Fața {i + 1} ---")

        # --- B. Pre-procesare (ca în Dataset) ---
        # 1. Decupăm fața din imaginea PIL
        face_crop_pil = original_pil_image.crop((x, y, x + w, y + h))

        # 2. Aplicăm transformările
        input_tensor = data_transform(face_crop_pil)

        # 3. Adăugăm o dimensiune de batch (1, C, H, W)
        input_tensor = input_tensor.unsqueeze(0).to(device)

        # --- C. Inferență (Predicție) ---
        with torch.no_grad():  # Oprim calculul gradienților
            outputs = model(input_tensor)

        # Extragem output-ul și îl mutăm pe CPU
        # Rezultatul este un tensor (1, 24) normalizat [0, 1]
        predicted_kpts_norm = outputs.cpu().numpy()[0]  # (24,)

        # Reshape la (12, 2) pentru (x, y)
        predicted_kpts_norm = predicted_kpts_norm.reshape(-1, 2)  # (12, 2)

        # --- D. Post-procesare (De-normalizare) ---
        # Inversăm normalizarea: din [0, 1] înapoi în coordonatele decupajului
        # Coordonata X = (x_norm * lățimea_decupajului)
        # Coordonata Y = (y_norm * înălțimea_decupajului)
        kpts_in_crop_coords = np.zeros_like(predicted_kpts_norm)
        kpts_in_crop_coords[:, 0] = predicted_kpts_norm[:, 0] * w
        kpts_in_crop_coords[:, 1] = predicted_kpts_norm[:, 1] * h

        # Adăugăm offset-ul decupajului pentru a obține coordonatele în imaginea ORIGINALĂ
        # Coordonata X = x_decupaj + x_colț_stânga_sus
        # Coordonata Y = y_decupaj + y_colț_stânga_sus
        kpts_in_original_coords = kpts_in_crop_coords + np.array([x, y])

        print("Coordonate Puncte Cheie (x, y):")

        # --- E. Afișare (Consolă și Imagine) ---
        # Desenăm BBox-ul feței
        cv2.rectangle(image_with_kpts, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Albastru

        for j, (px, py) in enumerate(kpts_in_original_coords):
            print(f"  Punct {j + 1}: ({px:.2f}, {py:.2f})")

            # Desenăm cercuri pe punctele cheie
            cv2.circle(
                image_with_kpts,
                (int(px), int(py)),
                radius=3,
                color=(0, 255, 0),  # Verde
                thickness=-1
            )
        print("=" * 30)

    # --- F. Plotare ---
    plt.figure(figsize=(16, 8))

    # Imaginea Originală
    plt.subplot(1, 2, 1)
    plt.title('Imagine Originală')
    plt.imshow(original_pil_image)  # PIL este RGB, deci e corect pentru plt
    plt.axis('off')

    # Imaginea cu Predicții
    plt.subplot(1, 2, 2)
    plt.title('Imagine cu Puncte Cheie (Ochi)')
    # Convertim BGR (OpenCV) în RGB (Matplotlib)
    plt.imshow(cv2.cvtColor(image_with_kpts, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()


# --- 5. RULARE ---
if __name__ == '__main__':
    # Verificăm dacă există modelul
    if not os.path.exists(MODEL_PATH):
        print(f"EROARE: Fișierul modelului '{MODEL_PATH}' nu a fost găsit.")
    # Verificăm dacă există imaginea
    elif not os.path.exists(IMAGE_TO_TEST_PATH):
        print(f"EROARE: Fișierul imaginii '{IMAGE_TO_TEST_PATH}' nu a fost găsit.")
        print("Modifică variabila 'IMAGE_TO_TEST_PATH' în script.")
    else:
        # Setăm dispozitivul
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Se folosește dispozitivul: {device}")

        # Inițializăm modelul
        model = get_model()

        # Încărcăm greutățile salvate
        print(f"Se încarcă greutățile modelului din: {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

        # Mutăm modelul pe dispozitiv
        model.to(device)

        # Setăm modelul în modul de evaluare (fără dropout, batchnorm etc.)
        model.eval()

        # Rulăm predicția
        predict_keypoints(IMAGE_TO_TEST_PATH, model, device)