import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Obtenir la résolution de l'écran
largeur, hauteur = pyautogui.size()

# Créer un arrière-plan blanc vide couvrant tout l'écran
arriere_plan = np.ones((hauteur, largeur, 3), dtype=np.uint8) * 255  # Fond blanc (255, 255, 255)

############## PARAMÈTRES #######################################################

# Définir ces valeurs pour afficher/masquer certains vecteurs de l'estimation
dessiner_regard = True
dessiner_axes_complets = True
dessiner_orientation_tete = False

# Multiplicateur de score de regard (un multiplicateur plus élevé = le regard affecte davantage l'estimation de l'orientation de la tête)
multiplicateur_score_x = 4
multiplicateur_score_y = 4

# Seuil de proximité des scores par rapport à la moyenne entre les images
seuil = 0.3

#################################################################################

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
    refine_landmarks=True,
    max_num_faces=2,
    min_detection_confidence=0.5)



#Pour demarer le projet avec une video enregistrée avec webcam standard 
#video_path = 'Enfant.mp4'  # Chemin vers votre vidéo
#cap = cv2.VideoCapture(video_path)

#Pour demarer le projet avec une video enregistrée 

cap = cv2.VideoCapture(0)




visage_3d = np.array([
    [0.0, 0.0, 0.0],            # Bout du nez
    [0.0, -330.0, -65.0],       # Menton
    [-225.0, 170.0, -135.0],    # Coin gauche de l'œil gauche
    [225.0, 170.0, -135.0],     # Coin droit de l'œil droit
    [-150.0, -150.0, -125.0],   # Coin gauche de la bouche
    [150.0, -150.0, -125.0]     # Coin droit de la bouche
    ], dtype=np.float64)

# Repositionner le coin gauche de l'œil comme origine
leye_3d = np.array(visage_3d)
leye_3d[:,0] += 225
leye_3d[:,1] -= 170
leye_3d[:,2] += 135

# Repositionner le coin droit de l'œil comme origine
reye_3d = np.array(visage_3d)
reye_3d[:,0] -= 225
reye_3d[:,1] -= 170
reye_3d[:,2] += 135

# Scores du regard de la frame précédente
dernier_lx, dernier_rx = 0, 0
dernier_ly, dernier_ry = 0, 0

while cap.isOpened():
    success, img = cap.read()
    cv2.imshow("Arrière-plan avec cercles", arriere_plan)

    # Inverser + convertir img de BGR à RGB
    img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

    # Pour améliorer les performances
    img.flags.writeable = False
    
    # Obtenir le résultat
    results = face_mesh.process(img)
    img.flags.writeable = True
    
    # Convertir l'espace couleur de RGB à BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    (hauteur_img, largeur_img, canaux_img) = img.shape
    visage_2d = []

    if not results.multi_face_landmarks:
      continue 

    for face_landmarks in results.multi_face_landmarks:
        visage_2d = []
        for idx, lm in enumerate(face_landmarks.landmark):
            # Convertir les coordonnées x et y des repères en coordonnées de pixels
            x, y = int(lm.x * largeur_img), int(lm.y * hauteur_img)
            
            # Ajouter les coordonnées 2D à un tableau
            visage_2d.append((x, y))
        
        # Obtenir les repères pertinents pour l'estimation de l'orientation de la tête
        visage_2d_tete = np.array([
            visage_2d[1],      # Nez
            visage_2d[199],    # Menton
            visage_2d[33],     # Coin gauche de l'œil gauche
            visage_2d[263],    # Coin droit de l'œil droit
            visage_2d[61],     # Coin gauche de la bouche
            visage_2d[291]     # Coin droit de la bouche
        ], dtype=np.float64)

        visage_2d = np.asarray(visage_2d)

        # Calculer le score de regard gauche en x
        if (visage_2d[243,0] - visage_2d[130,0]) != 0:
            score_lx = (visage_2d[468,0] - visage_2d[130,0]) / (visage_2d[243,0] - visage_2d[130,0])
            if abs(score_lx - dernier_lx) < seuil:
                score_lx = (score_lx + dernier_lx) / 2
            dernier_lx = score_lx

        # Calculer le score de regard gauche en y
        if (visage_2d[23,1] - visage_2d[27,1]) != 0:
            score_ly = (visage_2d[468,1] - visage_2d[27,1]) / (visage_2d[23,1] - visage_2d[27,1])
            if abs(score_ly - dernier_ly) < seuil:
                score_ly = (score_ly + dernier_ly) / 2
            dernier_ly = score_ly

        # Calculer le score de regard droit en x
        if (visage_2d[359,0] - visage_2d[463,0]) != 0:
            score_rx = (visage_2d[473,0] - visage_2d[463,0]) / (visage_2d[359,0] - visage_2d[463,0])
            if abs(score_rx - dernier_rx) < seuil:
                score_rx = (score_rx + dernier_rx) / 2
            dernier_rx = score_rx

        # Calculer le score de regard droit en y
        if (visage_2d[253,1] - visage_2d[257,1]) != 0:
            score_ry = (visage_2d[473,1] - visage_2d[257,1]) / (visage_2d[253,1] - visage_2d[257,1])
            if abs(score_ry - dernier_ry) < seuil:
                score_ry = (score_ry + dernier_ry) / 2
            dernier_ry = score_ry

        # Matrice de caméra
        focale = largeur_img
        mat_cam = np.array([ [focale, 0, hauteur_img / 3],
                             [0, focale, largeur_img / 3],
                             [0, 0, 1]])

        # Coefficients de distorsion
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Résoudre PnP
        _, l_rvec, l_tvec = cv2.solvePnP(leye_3d, visage_2d_tete, mat_cam, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        _, r_rvec, r_tvec = cv2.solvePnP(reye_3d, visage_2d_tete, mat_cam, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # Obtenir la matrice de rotation à partir du vecteur de rotation
        l_rmat, _ = cv2.Rodrigues(l_rvec)
        r_rmat, _ = cv2.Rodrigues(r_rvec)

        # Ajuster le vecteur d'orientation de la tête avec le score de regard
        l_gaze_rvec = np.array(l_rvec)
        l_gaze_rvec[2][0] -= (score_lx - 0.5) * multiplicateur_score_x
        l_gaze_rvec[0][0] += (score_ly - 0.5) * multiplicateur_score_y

        r_gaze_rvec = np.array(r_rvec)
        r_gaze_rvec[2][0] -= (score_rx - 0.5) * multiplicateur_score_x
        r_gaze_rvec[0][0] += (score_ry - 0.5) * multiplicateur_score_y

        # --- Projection ---

        # Obtenir le coin gauche de l'œil en tant qu'entier
        coin_gauche = visage_2d_tete[2].astype(np.int32)
        
        # Obtenir le coin droit de l'œil en tant qu'entier
        coin_droit = visage_2d_tete[3].astype(np.int32)

        # Projeter l'axe de rotation pour l'œil gauche
        axe = np.float32([[-100, 0, 0], [0, 100, 0], [0, 0, 300]]).reshape(-1, 3)
        axe_gauche, _ = cv2.projectPoints(axe, l_rvec, l_tvec, mat_cam, dist_coeffs)
        axe_regard_gauche, _ = cv2.projectPoints(axe, l_gaze_rvec, l_tvec, mat_cam, dist_coeffs)
        
        # Projeter l'axe de rotation pour l'œil droit
        axe_droit, _ = cv2.projectPoints(axe, r_rvec, r_tvec, mat_cam, dist_coeffs)
        axe_regard_droit, _ = cv2.projectPoints(axe, r_gaze_rvec, r_tvec, mat_cam, dist_coeffs)

        # Dessiner les axes de rotation projetés sur l'image
        cv2.line(img, coin_gauche, tuple(np.ravel(axe_regard_gauche[0]).astype(np.int32)), (0,0,0), 2)
        cv2.line(img, coin_gauche, tuple(np.ravel(axe_regard_gauche[1]).astype(np.int32)), (255,255,255), 2)
        cv2.line(img, coin_gauche, tuple(np.ravel(axe_regard_gauche[2]).astype(np.int32)), (0,0,255), 2)

        cv2.line(img, coin_droit, tuple(np.ravel(axe_regard_droit[0]).astype(np.int32)), (0,0,0), 2)
        cv2.line(img, coin_droit, tuple(np.ravel(axe_regard_droit[1]).astype(np.int32)), (255,255,255), 2)
        cv2.line(img, coin_droit, tuple(np.ravel(axe_regard_droit[2]).astype(np.int32)), (0,0,255), 2)

    cv2.imshow('suivi_regard_approche2', img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

