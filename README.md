# cs4342-final-project

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Clean Data
python clean_data.py

# Train Model
python train.py

** Might have to run the following before training based on your computer **
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")

# Evaluate Model
python evaluate.py

# Visualize Training Progress
python plot_training.py

# PCA Visualiztions
python visualize_pca.py

# Predict Your Own Images

## Single image
python predict.py path/to/outfit.jpg

## Multiple images
python predict.py image1.jpg image2.jpg image3.jpg

## Save results to file
python predict.py *.jpg --output predictions.json

## Show top-5 predictions
python predict.py image.jpg --top-k 5