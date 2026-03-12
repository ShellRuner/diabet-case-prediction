from fastapi import FastAPI, Request
from pydantic import BaseModel 
import pandas as pd
import joblib
from fastapi.templating import Jinja2Templates
import uvicorn


app = FastAPI()

# This tells FastAPI where your HTML files are stored
templates = Jinja2Templates(directory="template")

# charger le model entrainé
modele = joblib.load('model/random_forest_model.pkl')


#validation de la donnee
class DonneesEntree(BaseModel):
    Pregnancies: float  # Nombre de grossesses
    Glucose: float  # Concentration plasmatique de glucose à 2 heures dans un test oral de tolérance au glucose
    BloodPressure: float  # Tension artérielle diastolique (mm Hg)
    SkinThickness: float  # Épaisseur du pli cutané du triceps (mm)
    Insulin: float  # Insuline sérique sur 2 heures (mu U/ml)
    BMI: float  # Indice de masse corporelle (poids en kg/(taille en m)^2)
    DiabetesPedigreeFunction: float  # Fonction de pedigree du diabète
    Age: float  # Âge (années)


#home
@app.get("/")
def home():
    
    return {'message' : 'Welcome to the diabetes prediction API'}



@app.post("/predict_API")
def predict(data: DonneesEntree):
    # Here you would implement the prediction logic using your trained model
    
    data_dict = data.model_dump()

    #transform la data en dataframe
    data_df = pd.DataFrame([data_dict])

    #Prediction sur les données d'entrée

    prediction = modele.predict(data_df)
    probability = modele.predict_proba(data_df)[:, 1]  # Probabilité de la classe positive (diabète)

    #presentation de la prediction

    resultat = data_dict.copy()
    resultat['prediction'] = int(prediction[0])  # Convertir la prédiction en entier (0 ou 1)
    resultat['probability_diabete'] = float(probability[0])  # Convertir la probabilité en float

    return resultat




# @app.post("/prediction")
# async def prediction(
#     request: Request, # On injecte l'objet Request ici
#     # Si c'est un formulaire, on récupère les champs individuellement 
#     # ou via request.form()
# ):
#     # Récupération des données du formulaire
#     form_data = await request.form()

#     try:
#         data_dict = {key: float(value) for key, value in form_data.items()}
#     except ValueError:
#         return {"error": "All inputs must be numeric"}
#     # data_dict = dict(form_data)

#     # Conversion en DataFrame
#     data_df = pd.DataFrame([data_dict])

#     # Logique de prédiction
#     prediction = modele.predict(data_df)
#     probability = modele.predict_proba(data_df)[:, 1]

#     return templates.TemplateResponse(
#         "home.html",
#         {
#             "request": request,  # DOIT être l'objet Request de FastAPI
#             "prediction_text": f"The prediction is {prediction[0]} with a probability of {probability[0]:.2f}"
#         }
#     )

# 1. This shows the empty form when you visit the URL
@app.get("/prediction")
async def show_prediction_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# 2. This handles the data when the user clicks "Submit"
@app.post("/prediction")
async def do_prediction(request: Request):
    form_data = await request.form()
    
    try:
        data_dict = {key: float(value) for key, value in form_data.items()}
        data_df = pd.DataFrame([data_dict])
        
        prediction = modele.predict(data_df)
        probability = modele.predict_proba(data_df)[:, 1]

        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "prediction_text": f"the prediction is {prediction[0]} with a probability of {probability[0]:.2f}"
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "home.html", 
            {"request": request, "prediction_text": f"Error: {str(e)}"}

        )
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)