
## Being too Close to the Customer: Evaluating Influencer Empathy through Multimodal Deep Learning of Text, Audio, and Image Data

## Overview
This repository contains the code and data used in the research paper "Being too Close to the Customer: Evaluating Influencer Empathy through Multimodal Deep Learning of Text, Audio, and Image Data". 

## Installation
Before running the scripts, ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- Pandas
- Numpy
- Pickle
- Statsmodels
- Scikit-learn

You can install these packages using pip:
```bash
pip install torch pandas statsmodels numpy scikit-learn
```
--

## Dataset
Due to GitHub's storage limitations, only the test data is displayed in this repository. The test data has been uploaded to Google Drive. You can access the dataset using the following link: [Google Drive Dataset](https://drive.google.com/drive/folders/1d97Ox0in0WNW5miQZZ-zCo5xwq7QEivM).


### Multimodal Features 
#### Vocal Features
- **Tool Used**: Covarep.
- **Features**: 74 dimensions.


#### Visual Features
- **Tool Used**: OpenFace 2.0.
- **Features**: 35 dimensions.


#### Verbal Features
- **Tool Used**: BERT.
- **Features**: 768 dimensions.




## Usage
To run the analysis, follow these steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/multimodal-influencer-empathy/Multimodal-Influencer-empathy.git.
   ```
2. Navigate to the cloned directory.
3. Run the main script:
   ```bash
   python A1_LSTM_Text_run_evaluation_pretrained_model.py
   python A2_LSTM_Audio_run_evaluation_pretrained_model.py
   python A3_LSTM_Image_run_evaluation_pretrained_model.py
   python A4_EF_LSTM_Text#Audio_run_evaluation_pretrained_model.py
   python A5_EF_LSTM_Text#Image_run_evaluation_pretrained_model.py
   python A6_EF_LSTM_Audio#Image_run_evaluation_pretrained_model.py
   python A7_EF_LSTM_Text#Audio#Image_run_evaluation_pretrained_model.py
   python A8_TFN_Text#Audio#Image_run_evaluation_pretrained_model.py
   python A9_LMF_Text#Audio#Image_run_evaluation_pretrained_model.py
   python A10_MFN_Text#Audio#Image_run_evaluation_pretrained_model.py
   python A11_GMFN_noG_run_evaluation_pretrained_model.py
   python A12_GMFN_noW_run_evaluation_pretrained_model.py
   python A13_GMFN_noM_run_evaluation_pretrained_model.py
   python A14_GMFN_run_evaluation_pretrained_model.py
   ```

## Code Structure
- `14 main models`: The main script that orchestrates the data loading,  pretrained model loading, and evaluation.

- `pretrained_model`: Contains 14 pretrained models.

## Model Description
The codebase includes 14 different models, each designed to handle various modalities and fusion techniques in the context of measuring influencer emapthy:

 1. **No Fusion + Unimodal Data, LSTM Models (3 Models):**
   - `A1_LSTM_Text.py`: LSTM model for text data.
   - `A2_LSTM_Audio.py`: LSTM model for audio data.
   - `A3_LSTM_Image.py`: LSTM model for image data.

2. **Full Fusion + Bimodal Data, Early Fusion LSTM Models (3 Models):**
   - `A4_EF_LSTM_Text#Audio.py`: Early Fusion LSTM for text and audio data.
   - `A5_EF_LSTM_Text#Image.py`: Early Fusion LSTM for text and image data.
   - `A6_EF_LSTM_Audio#Image.py`: Early Fusion LSTM for audio and image data.

3. **Full Fusion + Trimodal Data (8 Models):**
   - `A7_EF_LSTM_Text#Audio#Image.py`: Early Fusion LSTM for text, audio and image data.
   - `A8_TFN_Text#Audio#Image.py`: TFN (Tensor Fusion Network) for text, audio and image data.
   - `A9_LMF_Text#Audio#Image.py`: LMF (Low-rank Multimodal Fusion) for text, audio and image data.
   - `A10_MFN_Text#Audio#Image.py`: MFN (Memory Fusion Network) for text, audio and image data.
   - `A11_GMFN_noG.py`: Based on Gragh MFN, the DFG (Dynamic Fusion Graph) component is removed, disabling the graph learning mechanism. To evaluate the contribution of the DFG component to the dynamic modality information fusion in the Gragh MFN model.
   - `A12_GMFN_noW.py`:Based on Gragh MFN, the dynamic temporal sequence features are eliminated, and only the static average features of each modality across the video duration are used. To validate the importance of temporal dynamics in Gragh MFN and compare their performance against static features.
   - `A13_GMFN_noM.py`: Based on Gragh MFN, the cross-modal interaction module is removed, meaning the interactions between text, audio, and image modalities are no longer captured. To assess the impact of cross-modal interaction mechanisms on the performance of the Gragh MFN model.
   - `A14_GMFN.py`: Graph MFN full model for text, audio and image data.


Each model is based on pretrained models and demonstrates the prediction results for different data modalities and fusion techniques. The models are specifically tailored for analyzing the influencer emapthy using multimodal data.



## Evaluation Metrics
The code includes functions for evaluating the model performance:
- Accuracy
- F1 Score
- Mean Absolute Error (MAE)
- Correlation Coefficient
- Loss






