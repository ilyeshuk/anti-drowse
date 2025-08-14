# anti-drowse
drowsiness detector (EAR/MAR/head tilt/idle) with PySide6 UI

# Anti-Drowse PRO — Local Drowsiness Detector (EAR/MAR/Head Tilt/Idle) with PySide6 UI

> Détection **locale** de somnolence au poste de travail : **yeux fermés** (EAR), **bâillement** (MAR), **tête penchée** (inclinaison latérale), **tête qui tombe vers l’avant** (proxy profondeur nez–yeux), **inactivité** souris/clavier.  
> Interface **PySide6**, **auto-calibration**, **snooze**, **son d’alarme** (beep ou .wav).  
> **100% local** – aucune vidéo ni audio n’est enregistrée ou envoyée.

---

## ✨ Fonctionnalités
- **Vision (MediaPipe Face Mesh)**
  - **EAR (Eye Aspect Ratio)** : détection de fermeture prolongée des yeux
  - **MAR (Mouth Aspect Ratio)** : détection de BAÎILLEMENTS répétés
  - **Inclinaison latérale (Roll)** : angle de la ligne des yeux (tête qui penche sur le côté)
  - **Bascule vers l’avant** : proxy via **Δz nez–yeux** (normalisé, baseline calibrée)
- **Inactivité clavier/souris** (via `pynput`)
- **Auto-calibration** (EAR/MAR/Δz) au démarrage (yeux ouverts ~7 s)
- **Interface PySide6** : seuils, frames, Idle, Snooze, sélection d’un **fichier .wav**
- **Alerte sonore** : beep intégré ou **.wav** personnalisé
- **Fusion des signaux** : alarme si l’un des critères dépasse ses seuils (yeux, bâillement, tilt, idle)
- **Snooze** configurable (désactive temporairement la détection après une alarme)

---

## 📦 Installation (recommandé dans un venv)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
