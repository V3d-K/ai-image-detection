# Basic AI vs Real Image Classification using Deep Learning

A small Convolutional Neural Network (CNN), built with TensorFlow/Keras, that classifies an image as **real** or **AI-generated**.


## Why this project?

Image generators such as GANs, Stable Diffusion and Midjourney now produce visuals that are hard to tell apart from human-made work. That matters for media verification, misinformation, intellectual property and content moderation. This project explores how far a lightweight, from-scratch CNN can go on that binary classification problem.

## Dataset

- **Source:** [AI Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images) on Kaggle (`cashbowman/ai-generated-images-vs-real-images`)
- **Size:** 950 images — 470 real, 480 AI-generated (roughly balanced)
- **Content:** web-scraped landscapes, paintings, portraits and more
- **Classes:** `RealArt` (human-made / photographed) and the AI-generated class folder

The dataset is **not** included in this repository (see [Getting the dataset](#getting-the-dataset)).

## Pipeline

1. **Download** the dataset via the Kaggle API.
2. **Resize** every image to 64×64 (Pillow) and skip unreadable files.
3. **Load** with `tf.keras.utils.image_dataset_from_directory` (labels inferred from folder names).
4. **Normalize** pixels from `[0, 255]` to `[0, 1]`.
5. **Split** 70 % train / 20 % validation / 10 % test.
6. **Train** the CNN for 20 epochs (Adam, binary cross-entropy), logging to TensorBoard.
7. **Plot** accuracy/loss curves and **save** the model.

## Model architecture

Input `64×64×3` → three `Conv2D(4×4, ReLU)` + `MaxPooling2D` blocks with 16 → 32 → 16 filters → `Flatten` → `Dense(32, ReLU)` → `Dense(1, sigmoid)`.

- **Parameters:** 30,081 (~117 KB)
- **Loss / optimizer:** `BinaryCrossentropy` / `Adam`

<p align="center">
  <img src="assets/model_summary.png" alt="Keras model summary" width="600">
</p>

## Results

| Metric (epoch 20) | Value |
|---|---|
| Training accuracy | ~99.8 % |
| Validation accuracy | ~94 % (94.27 % in the report, 94.79 % in the slides) |
| Training loss | ~0.01 |
| Validation loss | ~0.22 |

Exact numbers vary slightly from run to run.

<p align="center">
  <img src="assets/training_curves.png" alt="Training and validation loss and accuracy curves" width="800">
</p>

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── train.py              # Full pipeline: preprocess → train → plot → save
├── assets/                   # Images used in this README
│   ├── model_summary.jpg
│   └── training_curves.jpg
└── docs/
    ├── Project_Report.pdf    # Full project report
```

## Getting started

### 1. Install dependencies

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Getting the dataset

1. Create a Kaggle API token (Kaggle → *Settings* → *API* → *Create New Token*) and place `kaggle.json` in `~/.kaggle/`.
2. Download and unzip:

```bash
kaggle datasets download -d cashbowman/ai-generated-images-vs-real-images
unzip -qq ai-generated-images-vs-real-images.zip
```

3. Arrange the images so there is one sub-folder per class, which is what `image_dataset_from_directory` expects:

```
data/
├── RealArt/        # real images
└── <AI class>/     # AI-generated images
```

> **Never commit `kaggle.json`.** It is already listed in `.gitignore`.

### 3. Train

`src/train.py` was written in Google Colab and still uses the Colab paths (`/content/data/...`). Either run it in Colab, or edit the `source` / `destination` paths near the top for your machine, then:

```bash
python src/train.py
```

This trains the model, shows the accuracy and loss plots, and saves `cnn_model.h5`.

> ⚠️ The resize step **overwrites the original images in place**. Run it on a copy of the dataset if you want to keep the originals.

To watch training live:

```bash
tensorboard --logdir logs
```

## Known limitations

Being upfront about these so the results are read in the right context:

- **Small dataset** (950 images) at **64×64** resolution; results may not transfer to other generators or to larger, higher-quality images.
- **Dataset split:** `image_dataset_from_directory` reshuffles on every pass by default, so applying `take`/`skip` afterwards can mix images between train/validation/test across epochs. Using `shuffle=False` or a fixed, non-reshuffling split (or `validation_split` with a `seed`) would make the evaluation stricter.
- **No data augmentation** is applied in the current code.
- **The held-out test set is created but not evaluated** in `train.py`; the reported metrics come from the validation set.
- The model overfits after roughly epoch 10 (training loss keeps falling while validation loss plateaus).

## Future work

- Fix the split / add a proper test-set evaluation (confusion matrix, precision, recall, F1)
- Data augmentation and higher input resolution
- Transfer learning (VGG16, ResNet, EfficientNet)
- Video support, metadata analysis, and real-time detection

## References

- [Kaggle dataset – AI Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images)
- [Classifying AI-Generated and Original Images Using a CNN (ResearchGate)](https://www.researchgate.net/publication/387727185_Classifying_AIGenerated_and_Original_Images_Using_a_Convolutional_Neural_Network_Algorithm)
- [IJIRT paper](https://ijirt.org/publishedpaper/IJIRT166462_PAPER.pdf)
- See `docs/Project_Report.pdf` for the full reference list.
