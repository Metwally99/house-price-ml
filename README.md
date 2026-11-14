# House Price Prediction

A reproducible regression project that estimates historical property prices per unit area in New Taipei City, Taiwan. It compares five PyTorch architectures with a scikit-learn mean baseline, selects the best model on validation data, and deploys the trained weights to a browser demo and a Flask API.

[Try the live demo](https://metwally99.github.io/house-price-ml/) · [Machine-readable evaluation](docs/evaluation.json) · [Training code](train_models.py)

## Data

The [UCI Real Estate Valuation dataset](https://doi.org/10.24432/C5J30W) contains 414 sales and six input features. [The original CSV](data/houses.csv) is included under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Dataset credit goes to I-Cheng Yeh (2018). The training script maps the source headers to shorter names and excludes the row ID.

| Feature | Meaning |
| --- | --- |
| Date | Transaction year as a decimal |
| Age | Property age in years |
| Distance | Distance to the nearest MRT station in meters |
| Stores | Count of nearby convenience stores |
| Latitude, longitude | Property coordinates in degrees |

The target is price per unit area, measured in **10,000 New Taiwan dollars per ping**. One ping is about 3.3 m².

## Training and evaluation

The seeded split contains 264 training, 67 validation, and 83 test records. Feature standardization and target normalization are fitted on the training split. The five architectures use Adam and mean squared error. Validation MAE determines which model to deploy. The test split is used only to report the selected model and the baseline.

| Model | Hidden units | Validation MAE |
| --- | --- | ---: |
| Mean predictor | — | 9.91 |
| A · Linear regression | — | 6.06 |
| **B · Shallow network** | **16** | **5.89** |
| C · Two hidden layers | 32 → 16 | 7.79 |
| D · Wide hidden layer | 64 | 7.60 |
| E · Three hidden layers | 64 → 32 → 16 | 8.51 |

| Held-out test (83 records) | MAE ↓ | RMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: |
| Mean predictor | 11.00 | 13.20 | −0.04 |
| **B · Shallow network** | **6.21** | **8.38** | **0.58** |

Model B reduced held-out MAE by **43.6%** relative to the mean baseline. The deeper models had higher validation MAE on this split. The dataset represents one district in 2012–2013. Random splits measure performance on similar historical observations. Forecasting in a new period or location would require temporal or geographic validation. A single held-out split also leaves variation across possible samples unmeasured.

## How the model works

- **Linear algebra.** A record is a vector of six standardized features. A dense layer multiplies that vector by a weight matrix and adds a bias vector. Model B maps six features to 16 hidden activations using ReLU, then maps those activations to one scalar.
- **Probability.** The baseline estimates expected price with the training mean. Under squared error, the conditional mean minimizes expected loss, which motivates using MSE to train a model that can vary its prediction with the input features.
- **Evaluation.** MAE is the average absolute error in the original target unit. RMSE gives larger errors more weight. R² compares squared error to the variability of the observed test prices.

## Reproduce the results

Use Python 3.11–3.13. From the repository root:

```bash
python -m pip install -r requirements.txt
python train_models.py
```

Training regenerates `docs/evaluation.json`, the model export in `docs/models.json`, and PyTorch state dictionaries in `models/`. Set `DATASET_PATH` to a compatible CSV with the same six features and price target, or `MODEL_DIR` to change the location of the state dictionaries.

## Run the deployment

The GitHub Pages demo loads the exported weights and runs preprocessing and prediction in the browser. The Flask API serves the same model export and includes a health endpoint. The committed weights let both run immediately.

```bash
docker build -t house-price-ml .
docker run --rm -p 127.0.0.1:8000:8000 house-price-ml
```

Open [http://localhost:8000](http://localhost:8000), or call the API with a JSON request:

```http
POST /predict
Content-Type: application/json

{"date":2012.917,"age":32,"distance":84.87882,"stores":10,"latitude":24.98298,"longitude":121.54024}
```

The response contains `price`, the model code, and the target unit. Set `model_type` to `A` through `E` to compare models. For a local run without Docker, install the requirements and run `python app.py`.
