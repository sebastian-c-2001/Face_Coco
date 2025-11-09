import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import copy

# --- 1. CONFIGURARE & PARAMETRI ---

TRAIN_JSON_PATH = 'face_train_75p.json'
VAL_JSON_PATH = 'face_test_25p.json'
IMAGE_DIR = r'C:\Users\Sebi\Desktop\DB_MLAV\train2017'

# Parametri de antrenament
NUM_EPOCHS = 100
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_WORKERS = 4
MODEL_SAVE_PATH = 'best_model_face_keypoints.pth'  # Vom salva cel mai bun model aici

# Parametri pentru LR Scheduler personalizat
LEARNING_RATE = 0.01  # LR inițial
MIN_LR = 0.0001  # LR minim
LR_PATIENCE = 10  # Epoci de așteptat înainte de a scădea LR


# --- 2. CLASA DATASET PERSONALIZATĂ ---

class FaceKeypointDataset(Dataset):
    """
    Încarcă imaginile, le decupează după 'bbox' (care este 'face_box')
    și normalizează punctele cheie (keypoints) în funcție de decupaj.
    """

    def __init__(self, json_path, image_dir, transform=None):
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.image_dir = image_dir
        self.transform = transform

        # Indexăm după adnotări (fețe), nu după imagini
        self.annotations = data['annotations']
        # Creăm un index rapid pentru a găsi calea imaginii după ID
        self.image_map = {img['id']: img['file_name'] for img in data['images']}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ann = self.annotations[idx]

        image_id = ann['image_id']
        file_name = self.image_map[image_id]
        img_path = os.path.join(self.image_dir, file_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            return None  # Va fi filtrat de collate_fn

        bbox = ann['bbox']  # [x, y, w, h]
        x_min, y_min, w, h = bbox
        if w <= 0 or h <= 0:
            return None  # BBox invalid

        x_max = x_min + w
        y_max = y_min + h
        face_crop = image.crop((x_min, y_min, x_max, y_max))

        # 5. Transformă punctele cheie (Keypoints)
        # Punctele originale sunt [x1, y1, v1...] relativ la imaginea MARE
        # Noi le vrem [x1_norm, y1_norm...] relativ la CROP

        keypoints = np.array(ann['keypoints']).reshape(-1, 3)  # (12, 3)
        xy_keypoints = keypoints[:, :2]  # Luăm doar (x, y)

        # 5a. Translatare: mută originea în colțul BBox-ului
        xy_keypoints[:, 0] -= x_min
        xy_keypoints[:, 1] -= y_min

        # 5b. Normalizare: scalează la [0, 1] în funcție de mărimea crop-ului
        # Adăugăm un mic epsilon pentru a evita împărțirea la zero dacă w sau h sunt 0
        xy_keypoints[:, 0] /= (w + 1e-6)
        xy_keypoints[:, 1] /= (h + 1e-6)

        if self.transform:
            image_tensor = self.transform(face_crop)

        # Flatten: (12, 2) -> (24)
        keypoints_tensor = torch.tensor(xy_keypoints.flatten(), dtype=torch.float32)

        return image_tensor, keypoints_tensor


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)


# --- 3. DEFINIREA MODELULUI ---

def get_model():
    print("Se încarcă modelul ResNet-18 pre-antrenat...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Modificăm ultimul strat (Fully Connected)
    num_features = model.fc.in_features
    num_outputs = 12 * 2  # 12 puncte (x, y)

    model.fc = nn.Linear(num_features, num_outputs)

    # MSELoss funcționează pe output-uri ne-limitate.
    # Modelul va învăța să scoată valori [0, 1] deoarece etichetele sunt normalizate.
    return model


# --- 4. BUCLA DE ANTRENAMENT ---

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Se folosește dispozitivul: {device}")

    # 1. Definirea Transformărilor
    data_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        # Statistici de normalizare standard ImageNet
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # 2. Crearea seturilor de date (Datasets)
    train_dataset = FaceKeypointDataset(
        json_path=TRAIN_JSON_PATH,
        image_dir=IMAGE_DIR,
        transform=data_transform
    )
    val_dataset = FaceKeypointDataset(
        json_path=VAL_JSON_PATH,
        image_dir=IMAGE_DIR,
        transform=data_transform
    )

    # 3. Crearea DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 4. Inițializarea Modelului, Loss-ului și Optimizatorului
    model = get_model()
    model = model.to(device)

    # Funcția de Loss: Mean Squared Error (standard pentru regresie)
    criterion = nn.MSELoss()
    # Metrica: Mean Absolute Error (mai intuitivă decât MSE)
    metric_mae = nn.L1Loss()

    # Inițializăm optimizatorul cu LR-ul de start
    current_lr = LEARNING_RATE
    optimizer = optim.Adam(model.parameters(), lr=current_lr)

    # Variabile pentru salvarea celui mai bun model și scheduler-ul personalizat
    best_val_loss = float('inf')
    best_loss_for_lr_schedule = float('inf')
    patience_counter = 0

    print("--- Începe Antrenamentul ---")

    # 5. Bucla de Antrenament
    for epoch in range(NUM_EPOCHS):

        # --- Faza de Antrenare ---
        model.train()
        train_loss = 0.0
        train_mae_metric = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoca {epoch + 1}/{NUM_EPOCHS} [Train] LR={current_lr:.1e}"):
            if inputs.nelement() == 0: continue  # Sare peste batch-uri goale (din collate_fn)
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_mae_metric += metric_mae(outputs, labels).item() * inputs.size(0)

        # --- Faza de Validare ---
        model.eval()
        val_loss = 0.0
        val_mae_metric = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoca {epoch + 1}/{NUM_EPOCHS} [Val]"):
                if inputs.nelement() == 0: continue
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_mae_metric += metric_mae(outputs, labels).item() * inputs.size(0)

        # --- Calcul și Afișare Statistici Epoch ---
        len_train_data = max(1, len(train_dataset))
        len_val_data = max(1, len(val_dataset))

        epoch_train_loss = train_loss / len_train_data
        epoch_train_mae = train_mae_metric / len_train_data
        epoch_val_loss = val_loss / len_val_data
        epoch_val_mae = val_mae_metric / len_val_data

        print(f"\nEpoca {epoch + 1}/{NUM_EPOCHS} | "
              f"Train Loss: {epoch_train_loss:.6f} | Train MAE: {epoch_train_mae:.6f} | "
              f"Val Loss: {epoch_val_loss:.6f} | Val MAE: {epoch_val_mae:.6f}")

        # --- (NOU) Salvarea celui mai bun model ---
        if epoch_val_loss < best_val_loss:
            print(f"  Val Loss s-a îmbunătățit ({best_val_loss:.6f} --> {epoch_val_loss:.6f}). Se salvează modelul...")
            best_val_loss = epoch_val_loss
            # Salvăm o copie a stării modelului
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, MODEL_SAVE_PATH)

        # --- (NOU) Logică custom LR Scheduler ---
        if epoch_val_loss < best_loss_for_lr_schedule:
            # Dacă loss-ul s-a îmbunătățit, resetăm răbdarea
            best_loss_for_lr_schedule = epoch_val_loss
            patience_counter = 0
        else:
            # Dacă loss-ul stagnează, creștem răbdarea
            patience_counter += 1

        # Verificăm dacă răbdarea a expirat
        if patience_counter >= LR_PATIENCE:
            print(f"  Răbdarea a expirat (după {LR_PATIENCE} epoci). Se scade Learning Rate-ul.")
            current_lr *= 0.1  # Împărțim la 10
            patience_counter = 0  # Resetăm contorul
            best_loss_for_lr_schedule = epoch_val_loss  # Începem să contorizăm de la acest nou loss

            if current_lr < MIN_LR:
                print(f"  Learning Rate minim atins ({MIN_LR}). Se oprește antrenamentul.")
                break  # Oprim bucla 'for epoch'

            # Actualizăm LR-ul în optimizator
            print(f"  Noul Learning Rate: {current_lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

    print(f"\n--- Antrenament Finalizat ---")
    print(f"Cel mai bun model a fost salvat în: {MODEL_SAVE_PATH} (Val Loss: {best_val_loss:.6f})")


if __name__ == '__main__':
    main()