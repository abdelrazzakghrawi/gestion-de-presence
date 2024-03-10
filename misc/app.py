# Importation des bibliothèques nécessaires
import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

# Configuration de Firebase avec les informations d'identification
cred = credentials.Certificate('C:/Users/MY PC/Desktop/projet_tuto/gestion-de-presence/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://gestion-dabsence-default-rtdb.firebaseio.com/",
    'storageBucket':"gestion-dabsence.appspot.com"
})

# Accès au bucket de stockage Firebase
bucket = storage.bucket()

# Initialisation de la capture vidéo depuis la webcam
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # largeur .... CAP_PROP_FRAME_WIDTH ---> 3
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # hauteur .... CAP_PROP_FRAME_HEIGHT ---> 4

# Chargement de l'image d'arrière-plan
imgBackground = cv2.imread("static/Files/Resources/background.png")

# Chemin pour les différents modes
folderModePath = "static/Files/Resources/Modes/"
modePathList = os.listdir(folderModePath)
imgModeList = []

# Chargement des images de modes depuis les fichiers
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Chargement des encodages de visages pour identifier si la personne est dans notre base de données ou non
file = open("EncodeFile.p", "rb")
encodeListKnownWithIds = pickle.load(file)
file.close()
encodedFaceKnown, studentIds = encodeListKnownWithIds

# Initialisation des variables
modeType = 0
id = -1
imgStudent = []
counter = 0

while True:
    # Capturer une image depuis la webcam
    success, img = capture.read()

    # Redimensionner l'image pour des raisons d'efficacité computationnelle
    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    # Convertir l'image redimensionnée en format RGB pour la reconnaissance faciale
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    # Détecter les emplacements des visages dans l'image actuelle
    faceCurrentFrame = face_recognition.face_locations(imgSmall)

    # Encoder les visages dans l'image actuelle pour la reconnaissance
    encodeCurrentFrame = face_recognition.face_encodings(imgSmall, faceCurrentFrame)

    # Superposer l'image de la webcam sur l'image d'arrière-plan
    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if faceCurrentFrame:
        # Itérer à travers chaque visage détecté dans le frame actuel
        for encodeFace, faceLocation in zip(encodeCurrentFrame, faceCurrentFrame):
            # Comparer les visages actuels avec les visages connus
            matches = face_recognition.compare_faces(
                encodedFaceKnown, encodeFace
            )  # Les visages connus du fichier pickle seront comparés avec le visage actuel

            # Calculer les distances entre les visages
            faceDistance = face_recognition.face_distance(encodedFaceKnown, encodeFace)

            # Sélectionner l'indice du visage avec la distance minimale
            matchIndex = np.argmin(faceDistance)

            # Extraire les coordonnées du visage
            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Définir les coordonnées du rectangle englobant (BoundingBox)
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1

            # Dessiner le rectangle autour du visage sur l'image d'arrière-plan
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
            # Si un visage est détecté et correspond à un visage connu
            if matches[matchIndex] == True:
                # Afficher un message indiquant la détection d'un visage connu
                # et stocker l'identifiant de l'étudiant correspondant
                id = studentIds[matchIndex]

                # Si c'est la première détection du visage connu
                if counter == 0:
                    cvzone.putTextRect(imgBackground, "Face Detected", (65, 200), thickness=2)
                    cv2.waitKey(1)
                    counter = 1
                    modeType = 1

            # Si le visage détecté ne correspond à aucun visage connu
            else:
                # Afficher un message indiquant la détection d'un visage inconnu
                cvzone.putTextRect(imgBackground, "Face Detected", (65, 200), thickness=2)
                cv2.waitKey(3)

                # Afficher un message indiquant qu'aucun visage n'a été trouvé
                cvzone.putTextRect(imgBackground, "Face Not Found", (65, 200), thickness=2)

                # Changer le mode à 4 (modeType 4) et réinitialiser le compteur
                modeType = 4
                counter = 0

                # Modifier l'image d'arrière-plan en fonction du nouveau mode
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        # Si le compteur n'est pas égal à zéro (il y a une détection en cours)
        if counter != 0:
            # Si c'est la première détection (counter est égal à 1)
            if counter == 1:
                # Obtenir les données de l'étudiant depuis la base de données
                studentInfo = db.reference(f"Students/{id}").get()

                # Obtenir l'image de l'étudiant depuis le stockage Firebase
                blob = bucket.get_blob(f"static/Files/Images/{id}.jpg")
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                # Mettre à jour les données de présence
                datetimeObject = datetime.strptime(
                    studentInfo["last_attendance_time"], "%Y-%m-%d %H:%M:%S"
                )
                secondElapsed = (datetime.now() - datetimeObject).total_seconds()

                # Si plus de 30 secondes se sont écoulées depuis la dernière présence
                if secondElapsed > 30:
                    # Mettre à jour les données de présence dans la base de données
                    ref = db.reference(f"Students/{id}")
                    studentInfo["total_attendance"] += 1
                    ref.child("total_attendance").set(studentInfo["total_attendance"])
                    ref.child("last_attendance_time").set(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    )
                else:
                    # Changer le mode à 3 (modeType 3) et réinitialiser le compteur
                    modeType = 3
                    counter = 0
                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

            # Si le mode n'est pas égal à 3 (modeType 3)
            if modeType != 3:
                # Si le compteur est compris entre 10 et 20
                if 10 < counter <= 20:
                    # Changer le mode à 2 (modeType 2)
                    modeType = 2

                # Modifier l'image d'arrière-plan en fonction du nouveau mode
                imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
                # Si le compteur est inférieur ou égal à 10
                if counter <= 10:
                    # Afficher différentes informations de l'étudiant sur l'image d'arrière-plan
                    cv2.putText(
                        imgBackground,
                        str(studentInfo["total_attendance"]),
                        (861, 125),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        imgBackground,
                        str(studentInfo["major"]),
                        (1006, 550),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        imgBackground,
                        str(id),
                        (1006, 493),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        imgBackground,
                        str(studentInfo["standing"]),
                        (910, 625),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (100, 100, 100),
                        1,
                    )
                    cv2.putText(
                        imgBackground,
                        str(studentInfo["year"]),
                        (1025, 625),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (100, 100, 100),
                        1,
                    )
                    cv2.putText(
                        imgBackground,
                        str(studentInfo["starting_year"]),
                        (1125, 625),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6,
                        (100, 100, 100),
                        1,
                    )

                    # Calculer la taille du texte
                    (w, h), _ = cv2.getTextSize(
                        str(studentInfo["name"]), cv2.FONT_HERSHEY_COMPLEX, 1, 1
                    )

                    # Calculer l'offset pour centrer le texte
                    offset = (414 - w) // 2

                    # Afficher le nom de l'étudiant sur l'image d'arrière-plan
                    cv2.putText(
                        imgBackground,
                        str(studentInfo["name"]),
                        (808 + offset, 445),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (50, 50, 50),
                        1,
                    )

                    # Redimensionner l'image de l'étudiant
                    imgStudentResize = cv2.resize(imgStudent, (216, 216))

                    # Afficher l'image de l'étudiant sur l'image d'arrière-plan
                    imgBackground[175:175 + 216, 909:909 + 216] = imgStudentResize

                # Incrémenter le compteur
                counter += 1

                # Si le compteur atteint 20, réinitialiser le compteur, le mode et les informations de l'étudiant
                if counter >= 20:
                    counter = 0
                    modeType = 0
                    studentInfo = []
                    imgStudent = []

                # Si le modeType est égal à 0, réinitialiser le mode et le compteur
                else:
                    modeType = 0
                    counter = 0

                # Afficher l'image d'arrière-plan avec les modifications
                cv2.imshow("Face Attendance", imgBackground)

                # Si la touche 'q' est pressée, sortir de la boucle infinie
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
