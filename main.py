from typing import Optional
from fastapi import FastAPI
from joblib import load
from DataModel import DataModel, Row
import pandas as pd
import sklearn
from sklearn.metrics import r2_score
import math

app = FastAPI()
model = load("assets/modelo.joblib")

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/integrantes")
def intgrantes():
   return "Juan Felipe Peña Criado (201426463), Juan Diego González Gómez (201911031) y Sergio Ramírez Vélez (201714577)"

@app.post("/predict")
def make_predictions(row: Row):
   df = pd.DataFrame(row.dict(), columns=row.dict().keys(), index=[0])
   df.columns = row.columns()
   result = model.predict(df)
   return result[0]

@app.post("/r2")
def calculate_r2(data: DataModel):
   dataDict = data.dict()
   dataModels = dataDict["predictores"]

   if len(dataModels) > 1:
      predictions = []
      for dataModel in dataModels:
         df = pd.DataFrame(dataModel, columns=dataModel.keys(), index=[0])
         df.columns = data.columns()
         result = model.predict(df)
         predictions.append(result[0])
      
      r2 = r2_score(dataDict["valores_esperados"], predictions)
      return r2
   return "Se necesitan mínimo 2 registros para calcular el R^2 del modelo"
