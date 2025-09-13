import pandas as pd
from datasets import load_dataset
import bert_model
import csv
def getSymptomDataset():
    ds1 = load_dataset("venetis/symptom_text_to_disease_mk3", split="train")
    ds1.set_format("pandas")
    df1 = ds1[:]
    ds2 = load_dataset("venetis/symptom_text_to_disease_mk4", split="train")
    ds2.set_format("pandas")
    df2 = ds2[:]
    ds3 = load_dataset("celikmus/symptom_text_to_disease_01", split="train")
    ds3.set_format("pandas")
    df3 = ds3[:]
    df = pd.concat([df1, df2, df3])
    return df

df_symptom = getSymptomDataset()

symptom_model, symptom_tokenizer = bert_model.getModel(df_symptom, 25, "symptom_pred_model1.keras")



def getPredictedSymptoms(tokenizer):
    symptoms = ["\\ufemotional pain",
                "hair fallinig out",
                "heart hurts",
                "infected wound",
                "foot ache",
                "shoulder pain",
                "injury from sports",
                "skin issue",
                "stomach ache",
                "knee pain",
                "joint pain",
                "hard to breathe",
                "head ache",
                "body feels weak",
                "feeling dizzy",
                "back pain",
                "open wound",
                "internal pain",
                "blurry vision",
                "acne",
                "muscle pain",
                "neck pain",
                "cough",
                "ear ache",
                "feeling cold"]
    pred_symptoms = []
    general_condition_string = input('Say your problem: ')
    condition_sentences = general_condition_string.split(".")

    for s in condition_sentences:
        processed_data = bert_model.prepare_data(s, tokenizer)
        result = bert_model.make_prediction(symptom_model, processed_data=processed_data, classes=symptoms)
        pred_symptoms.append(result)

    return pred_symptoms

pred_symptoms = getPredictedSymptoms(symptom_tokenizer)

def match_symptoms(pred_symptoms):
    symptoms_match = []
    with open("symptom matching.csv") as f:
        reader = csv.reader(f)
        for line in reader:
            for p_s in pred_symptoms:
                if line[0] == p_s:
                    for s in line[1:]:
                        if s != "":
                            symptoms_match.append(s)
    return symptoms_match

symptoms_match = match_symptoms(pred_symptoms)

def getNBModel():
    trainNB = pd.read_csv("/content/drive/MyDrive/SIH/Training.csv")
    X_train_NB = trainNB.iloc[:, :-1]
    y_train_NB = trainNB["prognosis"]
    from sklearn.naive_bayes import MultinomialNB
    disease_model = MultinomialNB()
    disease_model.fit(X_train_NB, y_train_NB)
    return disease_model

disease_model = getNBModel()

def getDiseasePrediction(model, data):
    disease = model.predict(data)
    prob = model.predict_proba([data]).max()
    return disease, prob


unique_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
def encodeSymptoms(symptoms):
  encoded_symptoms = []
  for i in unique_symptoms:
    if i in symptoms_match:
      encoded_symptoms.append(1)
    else:
      encoded_symptoms.append(0)
  return encoded_symptoms

encoded_symptoms = encodeSymptoms(pred_symptoms)

pred_disease, pred_prob = getDiseasePrediction(disease_model, encoded_symptoms)
print(pred_disease, pred_prob)



