import os
from flask import Flask, request, render_template, jsonify
import joblib
from PIL import Image
import numpy as np
import xgboost as xgb
# Import TensorFlow/Keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3, EfficientNetV2M
from tensorflow.keras.layers import GlobalAveragePooling2D # Penting!
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Path ke model XGBoost kamu
MODEL_PATH = 'best_model.joblib'
IMG_SIZE = (299, 299) # Ukuran input yang diharapkan oleh InceptionV3 dan EfficientNetV2M
CLASSES = {
    0: 'Glioma',
    1: 'No Tumor',
    2: 'Meningioma',
    3: 'Pituitary'
}

# --- MUAT MODEL EKSTRAKTOR FITUR DARI PELATIHAN ANDA ---
eff_model_extractor = None
inc_model_extractor = None

try:
    # EfficientNetV2M feature extractor
    # Sesuai dengan kode pelatihan Anda
    base_eff_model = EfficientNetV2M(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    eff_output = GlobalAveragePooling2D()(base_eff_model.output)
    eff_model_extractor = Model(inputs=base_eff_model.input, outputs=eff_output)
    eff_model_extractor.trainable = False # Pastikan ini juga dibekukan
    print("EfficientNetV2M feature extractor berhasil dimuat dan dibekukan.")

    # InceptionV3 feature extractor
    # Sesuai dengan kode pelatihan Anda
    base_inc_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    inc_output = GlobalAveragePooling2D()(base_inc_model.output)
    inc_model_extractor = Model(inputs=base_inc_model.input, outputs=inc_output)
    inc_model_extractor.trainable = False # Pastikan ini juga dibekukan
    print("InceptionV3 feature extractor berhasil dimuat dan dibekukan.")

except Exception as e:
    print(f"Gagal memuat model CNN ekstraktor fitur: {e}")
    eff_model_extractor = None
    inc_model_extractor = None
# --- AKHIR MUAT MODEL EKSTRAKTOR FITUR ---


# --- FUNGSI EKSTRAKSI FITUR DUAL INPUT YANG AKURAT SESUAI PELATIHAN ---
def extract_dual_features_from_image(image_array_299x299):
    if eff_model_extractor is None or inc_model_extractor is None:
        raise RuntimeError("Model CNN ekstraktor fitur belum dimuat. Tidak bisa mengekstrak fitur.")

    # Pra-pemrosesan citra untuk EfficientNetV2M
    # Model Keras applications biasanya mengharapkan batch dimension (axis=0)
    # dan nilai piksel dalam rentang 0-255 untuk fungsi preprocess_input mereka.
    processed_image_efficientnet = efficientnet_preprocess(np.expand_dims(image_array_299x299, axis=0))
    # Ekstraksi fitur dari EfficientNetV2M
    eff_features = eff_model_extractor.predict(processed_image_efficientnet)
    # Output dari GlobalAveragePooling2D sudah 1D, jadi tidak perlu .flatten()
    # Tapi kita ambil elemen pertama karena predict mengembalikan batch (1, num_features)
    eff_features = eff_features[0]

    # Pra-pemrosesan citra untuk InceptionV3
    processed_image_inception = inception_preprocess(np.expand_dims(image_array_299x299, axis=0))
    # Ekstraksi fitur dari InceptionV3
    inc_features = inc_model_extractor.predict(processed_image_inception)
    # Output dari GlobalAveragePooling2D sudah 1D
    inc_features = inc_features[0]

    # Debugging: Cetak ukuran fitur untuk verifikasi
    print(f"Ukuran fitur EfficientNetV2M: {eff_features.shape}") # Akan mencetak (1280,)
    print(f"Ukuran fitur InceptionV3: {inc_features.shape}")     # Akan mencetak (2048,)

    # Gabungkan fitur dari kedua model
    # PENTING: Urutan penggabungan harus SAMA persis seperti saat pelatihan.
    # Jika Anda menggunakan np.concatenate([eff_features_train, inc_features_train], axis=1)
    # maka di sini juga harus eff_features dulu baru inc_features.
    # Kode pelatihan Anda menunjukkan eff_features_train dulu, jadi kita ikuti itu.
    combined_features = np.concatenate([eff_features, inc_features])

    print(f"Ukuran fitur gabungan: {combined_features.shape}") # Seharusnya (3328,)
    return combined_features

# --- AKHIR FUNGSI EKSTRAKSI FITUR ---


# --- MUAT MODEL XGBOOST UTAMA ---
xgb_classifier_model = None # Ganti nama variabel untuk menghindari konflik
try:
    xgb_classifier_model = joblib.load(MODEL_PATH)
    print(f"Model {MODEL_PATH} (XGBoost Classifier) berhasil dimuat.")
    if not hasattr(xgb_classifier_model, 'predict'):
        raise AttributeError("Model Joblib tidak memiliki metode 'predict'. Pastikan ini adalah model yang dapat memprediksi.")
    if not hasattr(xgb_classifier_model, 'predict_proba'):
        print("Peringatan: Model Joblib tidak memiliki metode 'predict_proba'. Probabilitas kelas tidak akan tersedia.")
except Exception as e:
    print(f"Gagal memuat model {MODEL_PATH}: {e}")
    # Jika model XGBoost gagal dimuat, biarkan 'xgb_classifier_model' tetap None


# --- ROUTE UTAMA DAN LOGIKA PREDIKSI ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_and_predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file bagian dalam request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    if file:
        try:
            # Baca gambar menggunakan PIL dan ubah ke RGB
            img = Image.open(file.stream).convert('RGB')

            # Ubah ukuran gambar ke 299x299
            img_resized = img.resize(IMG_SIZE)
            img_array = np.array(img_resized) # Konversi ke NumPy array (0-255)

            # Panggil fungsi ekstraksi fitur dual input
            extracted_features = extract_dual_features_from_image(img_array)

            # Ubah menjadi 2D array (jumlah_sampel, jumlah_fitur) untuk XGBoost
            # Karena hanya ada satu sampel, ini akan menjadi (1, 3328)
            input_for_xgboost = extracted_features.reshape(1, -1)

            # Prediksi menggunakan model XGBoost
            if xgb_classifier_model:
                if hasattr(xgb_classifier_model, 'predict_proba'):
                    probabilities = xgb_classifier_model.predict_proba(input_for_xgboost)[0] # Ambil probabilitas untuk sampel pertama
                    predicted_class_idx = np.argmax(probabilities)
                else:
                    # Jika model hanya memiliki predict (mengembalikan kelas langsung)
                    predicted_class_idx = xgb_classifier_model.predict(input_for_xgboost)[0]
                    # Probabilitas tidak akan tersedia, berikan dummy atau sesuaikan UI
                    probabilities = [0.0] * len(CLASSES)
                    if predicted_class_idx in CLASSES:
                        probabilities[predicted_class_idx] = 1.0

                predicted_class_label = CLASSES.get(predicted_class_idx, "Tidak dikenal")
                probability_of_predicted_class = float(probabilities[predicted_class_idx]) * 100

                # Format probabilitas untuk semua kelas
                class_probabilities_formatted = {CLASSES[i]: f"{p*100:.2f}%" for i, p in enumerate(probabilities)}

                return jsonify({
                    'success': True,
                    'prediction': predicted_class_label,
                    'probability': f"{probability_of_predicted_class:.2f}%",
                    'class_probabilities': class_probabilities_formatted
                })
            else:
                return jsonify({'error': 'Model klasifikasi (XGBoost) belum dimuat atau gagal dimuat.'}), 500

        except Exception as e:
            return jsonify({'error': f'Terjadi kesalahan saat memproses file atau memprediksi: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)