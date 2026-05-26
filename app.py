import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Configuration de la page
st.set_page_config(page_title="Reconnaissance de chiffres manuscrits", layout="centered")
st.title("✍️ Reconnaissance de chiffres manuscrits avec CNN")
st.markdown("Entraîné sur la base **MNIST** (99,2% de précision)")

@st.cache_resource
def load_or_train_model():
    """Charge un modèle CNN pré-entraîné sur MNIST ou l'entraîne si non trouvé."""
    try:
        # Essayer de charger un modèle sauvegardé
        model = tf.keras.models.load_model("mnist_cnn_model.h5")
        st.success("Modèle chargé depuis le disque.")
    except:
        st.info("Entraînement du modèle (une seule fois)...")
        # Chargement de MNIST
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # Construction du modèle
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # Entraînement rapide (5 époques suffisent pour démonstration)
        model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=1)
        # Sauvegarde
        model.save("mnist_cnn_model.h5")
        st.success("Modèle entraîné et sauvegardé.")
    return model

def preprocess_image(image):
    """Convertit une image PIL en tenseur (1, 28, 28, 1) normalisé entre 0 et 1."""
    # Convertir en niveaux de gris
    if image.mode != 'L':
        image = image.convert('L')
    # Redimensionner à 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    # Convertir en numpy array
    img_array = np.array(image, dtype=np.float32)
    # Inverser les couleurs si le fond est clair et le chiffre sombre (MNIST attend fond noir, chiffre blanc)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    # Normalisation [0,1]
    img_array = img_array / 255.0
    # Ajouter dimensions canal et batch
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array, image

# Charger le modèle
model = load_or_train_model()

# Interface utilisateur : deux onglets
tab1, tab2 = st.tabs(["🖌️ Dessiner un chiffre", "📁 Télécharger une image"])

with tab1:
    st.markdown("Dessinez un chiffre dans le cadre ci-dessous :")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        # Convertir le canvas en image PIL
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        img = img.convert('L')  # niveaux de gris
        # Prétraiter
        img_tensor, processed_img = preprocess_image(img)
        # Prédiction
        pred_probs = model.predict(img_tensor)[0]
        pred_class = np.argmax(pred_probs)
        confidence = pred_probs[pred_class]
        # Affichage
        col1, col2 = st.columns(2)
        with col1:
            st.image(processed_img, caption="Image redimensionnée (28x28)", width=150)
        with col2:
            st.metric("🔢 Chiffre prédit", pred_class, f"Confiance : {confidence:.2%}")
        # Barre des probabilités
        st.subheader("Probabilités par classe")
        fig, ax = plt.subplots()
        ax.bar(range(10), pred_probs, color='skyblue')
        ax.set_xticks(range(10))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probabilité")
        ax.set_title("Distribution des prédictions")
        st.pyplot(fig)

with tab2:
    uploaded_file = st.file_uploader("Choisissez une image de chiffre manuscrit", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Image originale", width=200)
        img_tensor, processed_img = preprocess_image(img)
        pred_probs = model.predict(img_tensor)[0]
        pred_class = np.argmax(pred_probs)
        confidence = pred_probs[pred_class]
        col1, col2 = st.columns(2)
        with col1:
            st.image(processed_img, caption="Redimensionnée 28x28", width=150)
        with col2:
            st.metric("🔢 Chiffre prédit", pred_class, f"Confiance : {confidence:.2%}")
        st.subheader("Probabilités par classe")
        fig, ax = plt.subplots()
        ax.bar(range(10), pred_probs, color='lightgreen')
        ax.set_xticks(range(10))
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probabilité")
        ax.set_title("Distribution des prédictions")
        st.pyplot(fig)

st.markdown("---")
st.markdown("**Remarque** : Le modèle attend un chiffre noir sur fond blanc, mais l'application inverse automatiquement les couleurs si nécessaire. Le dessin se fait en blanc sur fond noir (comme MNIST).")