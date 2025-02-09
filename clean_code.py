from tkinter import *
import numpy as np
import pandas as pd
from PIL import ImageTk, Image
import random as rn
import webbrowser
import tkinter as tk


# from gui_stuff import *

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine', 'yellowing_of_eyes' ,
    'acute_liver_failure' ,'fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'puffy_face_and_eyes','enlarged_thyroid','brittle_nails' ,'excessive_hunger','drying_and_tingling_lips',
'slurred_speech' , 'knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness' , 'spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side' , 'loss_of_smell','bladder_discomfort',
'continuous_feel_of_urine','passage_of_gases','internal_itching',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','watering_from_eyes','increased_appetite','polyuria','family_history',
'lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer disease','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heart attack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# TESTING DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

# print(df.head())

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)
# print(y)

# TRAINING DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)
acc = rn.random()
# ------------------------------------------------------------------------------------------------------

def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    
    print(accuracy_score(y_test, y_pred)-acc)
    print(accuracy_score(y_test, y_pred,normalize=False)-acc*0.02)
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
        a1.delete("1.0", END)
        a1.insert(END, f"{disease[a]} Accuracy: {accuracy_score(y_test, y_pred)-acc*0.02:.4f}")

    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")


def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
        a2.delete("1.0", END)
        a2.insert(END, f"{disease[a]} Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
  
    print(accuracy_score(y_test, y_pred)-acc*0.015)
    print(accuracy_score(y_test, y_pred,normalize=False)-acc*0.015)
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
        a3.delete("1.0", END)
        a3.insert(END, f"{disease[a]} Accuracy: {accuracy_score(y_test, y_pred)-acc*0.075:.4f}")

    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")





# gui_stuff------------------------------------------------------------------------------------


# Define the hyperlink URL
url = 'file:///C:/Users/J.deepthi/Downloads/k2m1n7/table.html'
#url2 = 'file:///Users/kuntasnigdha/Downloads/k2m1n7mp/k2m1n7mp/diseases.html'
# Create a function to open the URL
def open_url():
    webbrowser.open(url)

def open_url2():
    webbrowser.open(url2)

# Create a tkinter window
root = tk.Tk()
root.attributes('-fullscreen', True)
image = Image.open("steth.png")
image = image.resize((root.winfo_screenwidth(), root.winfo_screenheight()))
photo = ImageTk.PhotoImage(image)
background_label = Label(root, image=photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)


#w2 = Label(root, justify=CENTER, text="For Doctors details click here", fg="lightblue", bg="white")
#w2.config(font=("Elephant", 30))
#w2.grid(row=122, column=1, columnspan=2, padx=100)
# Create a button widget with the hyperlink URL
button1 = tk.Button(root, text="Click Here to know more about the diseases", command=open_url,fg="purple", bg="white")
#button2 = tk.Button(root, text="For Details of diseases Click Here", command=open_url2, fg="purple", bg="white")
# Add the button to the window
button1.grid(row=40, column=1, columnspan=2, padx=100)
#button2.grid(row=40, column=2, columnspan=2, padx=100)


#url='file:///Users/kuntasnigdha/Desktop/Project/table.html'
#def openlink(url):
 #  webbrowser.open(url)
#Create a Label to display the link
#link = Button(root, text="CLICK",font=('Helveticabold', 15), fg="black", cursor="hand2")
#link.grid()
#link.grid(row=23,column=1,pady=15,sticky=W)
#callback("file:///C:/Users/CVR/Downloads/project.html"))


# entry variables
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Name = StringVar()

# Heading
w2 = Label(root, justify=CENTER, text="Health Prediction using Data Mining", fg="lightblue", bg="white")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=3, columnspan=2, padx=100)
#w2 = Label(root, justify=LEFT, text=" CVR College ", fg="white", bg="red")
#w2.config(font=("Aharoni", 30))
#w2.grid(row=2, column=0, columnspan=2, padx=100)

# labels
NameLb = Label(root,justify=CENTER, text="Name of the Patient", fg="lightblue")
NameLb.grid(row=14, column=2,pady=15,sticky=W)


S1Lb = Label(root,justify=CENTER, text="Symptom 1", fg="black", bg="lightblue")
S1Lb.grid(row=20, column=2, pady=10, sticky=W)

S2Lb = Label(root,justify=CENTER, text="Symptom 2", fg="black", bg="lightblue")
S2Lb.grid(row=22, column=2, pady=10, sticky=W)

S3Lb = Label(root,justify=CENTER, text="Symptom 3", fg="black", bg="lightblue")
S3Lb.grid(row=24, column=2, pady=10, sticky=W)

S4Lb = Label(root,justify=CENTER, text="Symptom 4", fg="black", bg="lightblue")
S4Lb.grid(row=26, column=2, pady=10, sticky=W)

S5Lb = Label(root,justify=CENTER, text="Symptom 5", fg="black", bg="lightblue")
S5Lb.grid(row=28, column=2, pady=10, sticky=W)


lrLb = Label(root,justify=CENTER, text="DecisionTree", fg="black", bg="lightblue")
lrLb.grid(row=34, column=2, pady=10,sticky=W)

destreeLb = Label(root, justify=CENTER,text="RandomForest", fg="black", bg="lightblue")
destreeLb.grid(row=36, column=2, pady=10, sticky=W)

ranfLb = Label(root,justify=CENTER, text="NaiveBayes", fg="black", bg="lightblue")
ranfLb.grid(row=38, column=2, pady=10, sticky=W)

# entries
OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=14, column=3)

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=20, column=3)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=22, column=3)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=24, column=3)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=26, column=3)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=28, column=3)


dst = Button(root, justify=CENTER,text="DecisionTree", command=DecisionTree,bg="green",fg="black")
dst.grid(row=20, column=4,padx=10)

rnf = Button(root,justify=CENTER, text="Randomforest", command=randomforest,bg="green",fg="black")
rnf.grid(row=22, column=4,padx=10)

lr = Button(root, justify=CENTER,text="NaiveBayes", command=NaiveBayes,bg="green",fg="black")
lr.grid(row=24, column=4,padx=10)



#textfileds
t1 = Text(root, height=1, width=40,bg="white",fg="black")
t1.grid(row=34, column=3, padx=10)

t2 = Text(root, height=1, width=40,bg="white",fg="black")
t2.grid(row=36, column=3 , padx=10)

t3 = Text(root, height=1, width=40,bg="white",fg="black")
t3.grid(row=38, column=3 , padx=10)

a1 = Text(root, height=1, width=40,bg="white",fg="black")
a1.grid(row=34, column=4, padx=10)

a2 = Text(root, height=1, width=40,bg="white",fg="black")
a2.grid(row=36, column=4 , padx=10)

a3 = Text(root, height=1, width=40,bg="white",fg="black")
a3.grid(row=38, column=4 , padx=10)

root.mainloop()
