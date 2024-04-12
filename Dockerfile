FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

COPY requirements.txt ./requirements.txt

COPY ./Model/decision_tree_reg_latest.pkl /app/Model/decision_tree_reg_latest.pkl
COPY ./Preprocessing_models/decision_tree_reg_latest.joblib /app/model/decision_tree_reg_latest.joblib
COPY ./Preprocessing_models/lr_model.joblib /app/model/lr_model.joblib
COPY ./Preprocessing_models/pca.joblib /app/model/pca.joblib
COPY ./Preprocessing_models/scaler.joblib /app/model/scaler.joblib

RUN pip install -r requirements.txt

EXPOSE 8000

ENV FLASK_APP=app.py

CMD ["flask" , "run", '--host=0.0.0.0' , "--port=8000"]