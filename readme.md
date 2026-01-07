# Satellite Imagery–Based Property Valuation
## A Multimodal Machine Learning Approach

# 1 Overview

This project develops a multimodal regression pipeline for residential property valuation
by combining traditional tabular housing attributes with satellite imagery. The core objective
is to incorporate environmental and neighborhood context—such as green cover, road density,
and urban layout—into pricing models.
The final system uses a hybrid CNN + XGBoost architecture, where convolutional
neural networks extract visual representations from satellite images and a tree-based model
performs calibrated price prediction.

# 2 Key Contributions

• Programmatic acquisition of satellite images using the Mapbox Static Images API
• CNN-based feature extraction from satellite imagery
• Multimodal learning using tabular and visual features
• Model explainability using Grad-CAM
• Hybrid CNN + XGBoost model outperforming tabular-only baselines

# 3 Repository Structure

data/
    train.csv / train.xlsx
    test.csv / test.xlsx
    images/
        train/
        test/
src/
    dataset.py
    model.py
    gradcam.py

preprocessing.ipynb
model_training.ipynb
data_fetcher.ipynb

outputs/
    best_model.pth
    final_predictions.csv

# 4 Environment Setup

## 4.1 Install Dependencies

pip install -r requirements . txt

4.2 Core Libraries
• PyTorch, torchvision
• scikit-learn
• XGBoost
• pandas, numpy
• OpenCV
• matplotlib, seaborn

# 5 Satellite Image Acquisition

Satellite images are downloaded using the Mapbox Static Images API.
 
## 5.1 Set API Token
 
export MAPBOX_TOKEN = your_api_key_here # Linux / Mac
set MAPBOX_TOKEN = your_api_key_here # Windows
 
## 5.2 Download Images
 
python src / data_fetcher . py
Images are saved under:
data/images/train/
data/images/test/

# 6 Data Preprocessing and EDA

Run the preprocessing notebook:
jupyter notebook notebooks / preprocessing . ipynb
This step includes:
• Price distribution analysis
• Log transformation of the target variable
• Feature scaling for tabular data
• Geospatial sanity checks

# 7 Multimodal CNN Training

The end-to-end multimodal model combines CNN image embeddings with tabular features.
jupyter notebook  model_training . ipynb
7.1 Training Details
• Target: log(1 + price)
• Loss: Mean Squared Error (MSE)
• Evaluation metric: RMSE (log-price)
• Early stopping based on validation RMSE
The best-performing model is saved as:
outputs/best_model.pth

# 8 Explainability with Grad-CAM

Grad-CAM is applied to the CNN backbone to visualize which image regions influence predic-
tions.
Highlighted patterns include:
• Green spaces and tree cover
• Road connectivity
• Building density
• Neighborhood structure
Grad-CAM visualizations are stored in:
outputs/gradcam/

# 9 Hybrid CNN + XGBoost Model

To maximize performance, CNN-extracted image embeddings are concatenated with tabular
features and passed to an XGBoost regressor.
jupyter notebook notebooks / hybrid_xgb . ipynb
This hybrid approach leverages:
• CNNs for representation learning
• Tree-based models for calibration and nonlinear interactions

# 10 Model Performance

Model RMSE (log) Approx. Price Error R2
Tabular Baseline 0.277  ∼32% 0.721
CNN + MLP 0.274 ∼31% 0.728
Hybrid CNN + XGBoost 0.179  ∼19.6% 0.883

# 11 Final Predictions

The final submission file is generated as:
outputs/final_predictions.csv
Format:
id,predicted_price
Predictions are converted back from log(1 + price) to the original price scale.

# How to Run the Project (QUICK START)
1. Download the dataset
2. Set Mapbox API token
3. Fetch satellite images
4. Run preprocessing
5. Train CNN
6. Train hybrid XGBoost
7. Generate final predictions
