# Skin Lesion Classification Using Deep Learning

**A Computer Vision and Deep Learning Project | Custom CNN | MobileNetV2 | Transfer Learning | LLaVA | Streamlit**

## 1. Project Overview

Skin lesion image classification is a challenging computer vision problem. Different skin lesion categories can have visually similar characteristics, while medical image datasets often contain an unequal number of images across categories.

The main objective of this project was to develop a deep learning-based image classification system that can learn visual patterns from dermatoscopic images and classify them into seven categories using the HAM10000 dataset.

Rather than relying on a single model, I implemented and compared two deep learning approaches:

- **Custom Convolutional Neural Network (CNN):** Designed and trained a CNN architecture from scratch to learn image features.
- **MobileNetV2 Transfer Learning:** Used an ImageNet-pretrained MobileNetV2 model, adapted it to the dataset, and performed fine-tuning.

I also explored Vision Language Model integration using LLaVA 7B through Ollama to generate general visual descriptions of uploaded images.

Finally, I developed an interactive Streamlit application that combines image classification and visual description generation.

**The project covers the complete machine learning workflow:** dataset exploration, preprocessing, model development, transfer learning, fine-tuning, evaluation, Vision Language Model integration, and application development.

## 2. Problem Statement

The HAM10000 dataset contains dermatoscopic images belonging to seven different skin lesion categories. Building a reliable classification model using this dataset involves several challenges:

1. **Multi-class classification:** The model must distinguish between seven different image categories.
2. **Class imbalance:** Some categories contain significantly more images than others, which can affect model performance.
3. **Visual similarity:** Different categories can have overlapping visual characteristics, making classification challenging.
4. **Data leakage:** Multiple images may belong to the same lesion. If related images appear in both training and testing data, evaluation results may become misleading.
5. **Model performance:** A custom CNN and a pretrained architecture may perform differently across individual classes.

### My Approach

To address these challenges, I:

- Explored the dataset and analysed its class distribution.
- Prepared image paths and processed images for model training.
- Used lesion-level grouping to reduce data leakage risk.
- Developed a custom CNN as a baseline model.
- Applied MobileNetV2 transfer learning and fine-tuning.
- Compared both models using multiple evaluation metrics.
- Analysed class-wise performance to identify weaknesses.
- Integrated LLaVA for general visual image descriptions.
- Built a Streamlit application to demonstrate the workflow.

## 3. Dataset Description

**Dataset:** HAM10000 – Human Against Machine with 10,000 training images.

The dataset contains approximately 10,015 dermatoscopic images across seven categories.

| Class Label | Description |
|---|---|
| akiec | Actinic keratoses and intraepithelial carcinoma |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

The metadata file contains important information such as:

- `image_id`: Unique image identifier.
- `dx`: Image category label.
- `lesion_id`: Identifier associated with the lesion.

**Dataset source:** [HAM10000 – Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T)

The dataset is not included in this repository. It must be downloaded separately and placed in the `data/` directory.

## 4. Project Implementation – Step by Step

The project was implemented using six Jupyter notebooks. Each notebook handles a specific stage of the machine learning workflow.

### Step 1: Dataset Exploration

**Notebook:** `Dataset_Exploration.ipynb`

First, I explored the HAM10000 dataset to understand its structure and identify potential data quality issues.

Work performed:

- Loaded the metadata CSV using Pandas.
- Inspected dataset dimensions, columns, and data types.
- Analysed the distribution of all seven lesion categories.
- Checked missing values and duplicate records.
- Explored image identifiers and lesion identifiers.
- Visualized sample images from different categories.
- Examined the imbalance between majority and minority classes.

**Why this step matters:** Understanding the dataset before training helps identify class imbalance, missing information, and potential issues that can affect model performance.

### Step 2: Data Preprocessing

**Notebook:** `Data_Preprocessing.ipynb`

After understanding the dataset, I prepared the images and labels for deep learning.

Work performed:

- Loaded the metadata and mapped image identifiers to their actual image files.
- Combined image locations from both HAM10000 image folders.
- Resized images to 224 × 224 pixels.
- Converted images into the required numerical format.
- Encoded the seven target categories.
- Prepared training, validation, and held-out testing data.
- Used lesion-level grouping to reduce the possibility of the same lesion appearing in multiple data partitions.

**Important data handling decision:**

The dataset contains multiple images associated with the same lesion. Therefore, lesion-level grouping was considered during data splitting to reduce data leakage and make model evaluation more meaningful.

**Why this step matters:** Consistent preprocessing and appropriate data splitting are essential for training and evaluating deep learning models fairly.

### Step 3: Custom CNN Development

**Notebook:** `CNN_Model.ipynb`

I developed a custom Convolutional Neural Network from scratch to establish a baseline for skin lesion image classification.

**Architecture:**

```text
Input Image (224 × 224 × 3)
          |
          v
Conv2D (32 filters)
          |
Batch Normalization
          |
ReLU Activation
          |
MaxPooling + Dropout
          |
          v
Conv2D (64 filters)
          |
Batch Normalization
          |
ReLU Activation
          |
MaxPooling + Dropout
          |
          v
Conv2D (128 filters)
          |
Batch Normalization
          |
ReLU Activation
          |
MaxPooling + Dropout
          |
          v
Global Average Pooling
          |
Dense Layer (128 units)
          |
Dropout
          |
          v
Softmax Output (7 Classes)
```

**Implementation details:**

- Used three convolutional blocks with 32, 64, and 128 filters.
- Applied Batch Normalization to improve training stability.
- Used ReLU activation to learn nonlinear features.
- Applied Max Pooling to reduce spatial dimensions.
- Added Dropout to help reduce overfitting.
- Used Global Average Pooling instead of a large flattening layer.
- Added a Dense layer with 128 units.
- Used a seven-unit Softmax output layer for multi-class classification.

**Training callbacks:**

- EarlyStopping – Stops training when validation loss stops improving.
- ModelCheckpoint – Saves the best-performing model checkpoint.
- ReduceLROnPlateau – Reduces the learning rate when validation loss stops improving.

**Why I built this model:** A custom CNN provides a baseline to understand how a model trained specifically on the dataset performs before introducing pretrained architectures.

### Step 4: Transfer Learning Using MobileNetV2

**Notebook:** `Transfer_Learning.ipynb`

After building the custom CNN, I implemented transfer learning using MobileNetV2 pretrained on ImageNet.

Instead of learning every visual feature from scratch, the model reuses pretrained visual representations and adapts them to the seven HAM10000 categories.

**Implementation process:**

1. Loaded MobileNetV2 with ImageNet pretrained weights.
2. Removed the original classification head using `include_top=False`.
3. Initially froze the pretrained base layers.
4. Applied MobileNetV2-specific image preprocessing.
5. Added Global Average Pooling.
6. Added Dropout for regularization.
7. Added a Dense Softmax output layer with seven units.
8. Trained the newly added classification head.
9. Performed fine-tuning to adapt pretrained features to the target dataset.
10. Saved the best model checkpoints.

**Why MobileNetV2?**

MobileNetV2 is a lightweight convolutional architecture designed to provide efficient feature extraction. Its pretrained weights offer a useful starting point for transfer learning.

**Objective:** Compare the performance of a custom CNN against a pretrained deep learning architecture.

### Step 5: Model Evaluation and Comparison

**Notebook:** `Model_Evaluation_1.ipynb`

After training both architectures, I evaluated them on the same held-out test set containing **2,014 images**.

I used multiple metrics rather than depending only on accuracy.

**Evaluation metrics:**

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Macro ROC-AUC (One-vs-Rest)
- Test Loss
- Class-wise F1-score
- Confusion Matrix

### Overall Model Performance

| Evaluation Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Accuracy | 74.03% | **76.27%** |
| Macro Precision | 45.69% | **51.94%** |
| Macro Recall | 41.20% | **47.09%** |
| Macro F1-score | 42.18% | **48.13%** |
| Macro ROC-AUC (OvR) | 90.41% | **92.22%** |
| Test Loss | 0.6905 | **0.6726** |

### Class-wise F1-score Comparison

| Class | Test Images | Custom CNN | MobileNetV2 |
|---|---:|---:|---:|
| akiec | 68 | **0.35** | 0.29 |
| bcc | 100 | 0.42 | **0.52** |
| bkl | 205 | 0.40 | **0.49** |
| df | 22 | 0.00 | 0.00 |
| mel | 222 | 0.40 | **0.44** |
| nv | 1,363 | 0.88 | **0.90** |
| vasc | 34 | 0.50 | **0.74** |

### What I Learned from the Results

The evaluation showed that MobileNetV2 performed better overall than the custom CNN.

Key observations:

- MobileNetV2 achieved 76.27% accuracy compared with 74.03% for the custom CNN.
- Macro F1-score improved from 42.18% to 48.13%.
- Macro recall improved from 41.20% to 47.09%.
- Macro ROC-AUC improved from 90.41% to 92.22%.
- MobileNetV2 achieved better F1-scores for bcc, bkl, mel, nv, and vasc.
- Custom CNN performed better on akiec.
- Both models achieved an F1-score of zero for df.

**Final model selection:** MobileNetV2 was selected as the stronger overall model based on the evaluated metrics and used in the Streamlit application.

**Important finding:** Although MobileNetV2 achieved better overall performance, the class-wise results revealed that minority categories remain challenging. This highlights why macro-averaged and class-wise evaluation are important for imbalanced datasets.

### Step 6: Vision Language Model Experiment

**Notebook:** `Vision_Language_models.ipynb`

In addition to image classification, I explored how a Vision Language Model can generate natural-language descriptions of image content.

**Model:** LLaVA 7B  
**Runtime:** Ollama  
**API:** Ollama local API

**Implementation process:**

1. Connected to the locally running Ollama service.
2. Checked the availability of the LLaVA model.
3. Selected test images for the experiment.
4. Converted image data into Base64 format.
5. Sent images and prompts to the Ollama generation API.
6. Generated descriptions focused on visible characteristics.
7. Saved experiment results into CSV files.
8. Extended the experiment to batch image description generation.

The prompts were designed to focus on general visible characteristics such as:

- Colours
- Patterns
- Texture
- Image quality

**Why I added this experiment:**

I wanted to explore the difference between a traditional image classification model and a Vision Language Model.

The classification model produces scores for predefined categories, whereas LLaVA generates natural-language descriptions of visual content.

These are separate tasks and their outputs should not be treated as interchangeable.

## 5. Streamlit Web Application

After developing and evaluating the models, I built an interactive web application using Streamlit.

The application combines the trained MobileNetV2 classifier and the LLaVA image-description experiment.

### Application Features

- Image upload through the Streamlit interface.
- Uploaded image preview.
- Image preprocessing before inference.
- MobileNetV2-based seven-class classification.
- Predicted dataset category and model score.
- Class-wise prediction score visualization.
- LLaVA-generated general visual description.
- Organized dashboard interface with project information and an educational disclaimer.

### Application Workflow

```text
       User Uploads Image
               |
               v
        Image Preview
               |
               v
        Image Preprocessing
               |
        +------+------+
        |             |
        v             v
   MobileNetV2      LLaVA 7B
   Classifier       via Ollama
        |             |
        v             v
  Class Scores     Text Description
        |             |
        +------+------+
               |
               v
       Streamlit Dashboard
```

The classification and description components operate independently. The LLaVA-generated description does not determine or validate the classifier's prediction.

## 6. Project Architecture

```text
Skin_Cancer_CNN/
│
├── app/
│   ├── app.py
│   ├── utils.py
│   └── style.css
│
├── data/
│   ├── HAM10000_metadata.csv
│   ├── HAM10000_images_part_1/
│   └── HAM10000_images_part_2/
│
├── notebooks/
│   ├── CNN_Model.ipynb
│   ├── Data_Preprocessing.ipynb
│   ├── Dataset_Exploration.ipynb
│   ├── Model_Evaluation_1.ipynb
│   ├── Transfer_Learning.ipynb
│   └── Vision_Language_models.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   └── model_utils.py
│
├── models/
│   ├── custom_cnn_best.keras
│   ├── mobilenetv2_best.keras
│   ├── mobilenetv2_finetuned_best.keras
│   └── skin_lesion_custom_cnn.keras
│
├── outputs/
│   └── Evaluation and VLM experiment results
│
├── requirements.txt
├── check_environment.py
└── README.md
```

### Explanation of Important Folders

| Folder | Purpose |
|---|---|
| `app/` | Streamlit application and inference utilities |
| `data/` | HAM10000 dataset and metadata |
| `notebooks/` | Complete experimentation and development workflow |
| `src/` | Reusable configuration, data, and model utilities |
| `models/` | Saved trained model checkpoints |
| `outputs/` | Evaluation results and experiment outputs |

## 7. Technologies and Tools

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Deep Learning | TensorFlow, Keras |
| Computer Vision | CNN, MobileNetV2 |
| Transfer Learning | ImageNet-pretrained MobileNetV2 |
| Model Evaluation | Scikit-learn |
| Image Processing | Pillow |
| Vision Language Model | LLaVA 7B |
| Local Model Runtime | Ollama |
| Web Application | Streamlit |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git |

## 8. Installation and Execution

### Prerequisites

- Python 3.11
- HAM10000 dataset
- Ollama installed locally
- LLaVA 7B available through Ollama

### Step 1: Clone or Download the Project

Download the project source code and open its root folder in a terminal.

### Step 2: Create a Virtual Environment

**Windows PowerShell:**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows CMD:**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Prepare the Dataset

Place the HAM10000 metadata and image folders inside `data/`:

```text
data/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

### Step 5: Verify Ollama and LLaVA

Check available Ollama models:

```bash
ollama list
```

Ensure that `llava:7b` is available and the Ollama service is running.

The application connects to:

```text
 http://localhost:8501
```

### Step 6: Launch the Application

From the project root:

```bash
streamlit run app/app.py
```

The Streamlit application will open in the browser.

## 9. Challenges and Learnings

During this project, I worked through several practical challenges:

**1. Imbalanced Dataset**

The seven categories had unequal image counts. I therefore examined class-wise F1-score and macro-averaged metrics instead of evaluating only accuracy.

**2. Data Leakage Prevention**

Multiple images can belong to the same lesion. Lesion-level grouping was considered during data splitting to reduce the risk of related images appearing in both training and test sets.

**3. Model Selection**

I implemented a custom CNN and MobileNetV2 rather than relying on a single architecture. Comparing their results helped me understand the benefits of transfer learning.

**4. Fine-Tuning**

I first used MobileNetV2 as a frozen feature extractor and then performed fine-tuning to adapt pretrained features to the target dataset.

**5. Integrating a Vision Language Model**

I connected LLaVA through Ollama and experimented with image-to-text generation using the local API.

**6. Application Development**

I combined image preprocessing, trained model inference, score visualization, and visual description generation into an interactive Streamlit application.

## 10. Limitations and Future Improvements

### Current Limitations

- The dataset has a significant class imbalance.
- The df category had an F1-score of zero for both evaluated models.
- Some minority categories had relatively few test samples.
- Overall accuracy does not fully represent performance across all categories.
- The model is limited to the seven categories represented in the dataset.
- LLaVA descriptions are generated text and are not verified medical information.
- Model outputs have not been validated for clinical use.

### Future Improvements

- Experiment with class-weighted loss functions.
- Explore targeted augmentation for underrepresented categories.
- Evaluate additional pretrained architectures.
- Perform systematic hyperparameter tuning.
- Explore Grad-CAM for visual model interpretability.
- Add experiment tracking and model versioning.
- Evaluate generalization using independent datasets.

## 11. Disclaimer

This project is developed strictly for educational and research purposes in deep learning and computer vision.

It is not a medical diagnostic system and must not be used to diagnose skin conditions, estimate medical risk, or make treatment decisions. Classification scores and generated descriptions are experimental outputs and are not a substitute for professional medical evaluation.

## 12. Project Summary

Through this project, I implemented an end-to-end deep learning workflow for multi-class skin lesion image classification.

My work included:

- Exploring and understanding the HAM10000 dataset.
- Preparing image data and labels for model training.
- Developing a custom CNN architecture.
- Implementing MobileNetV2 transfer learning and fine-tuning.
- Evaluating and comparing models using multiple metrics.
- Analysing class-wise performance and model limitations.
- Integrating LLaVA 7B through Ollama for general image descriptions.
- Building an interactive Streamlit application.

**Key result:** MobileNetV2 achieved 76.27% accuracy and a 48.13% macro F1-score on the held-out test set of 2,014 images, outperforming the custom CNN on the overall evaluation metrics.

**Core Skills Demonstrated:** Python | Deep Learning | CNN | TensorFlow | Keras | Computer Vision | Transfer Learning | MobileNetV2 | Model Evaluation | LLaVA | Ollama | Streamlit