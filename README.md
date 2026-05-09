Project: MarketPulse ANN + MLOps

Steps to run:
1. pip install -r requirements.txt
2. python src/ingestion/data_ingestion.py
3. python src/preprocessing/sentiment.py
4. python src/preprocessing/dataset_builder.py
5. python src/models/lstm_gru_model.py

Run API:
uvicorn app:app --reload

Run Docker:
docker build -t marketpulse-api .
docker run -p 8000:8000 marketpulse-api