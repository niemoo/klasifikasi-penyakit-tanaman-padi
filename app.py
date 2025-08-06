import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_padi.h5')

model = load_model()

# Kelas prediksi
class_names = ['blast', 'blight', 'Bukan termasuk tanaman padi', 'sehat', 'tungro']

# Deskripsi untuk setiap kelas
class_description = {
    'blast': 'Penyakit pada padi yang disebabkan oleh jamur Pyricularia oryzae ditandai bercak berbentuk belah ketupat pada daun. Penyakit ini dapat ditemukan pada beberapa bagian tanaman padi seperti daun, leher daun, batang, malai dan biji. Selain itu, penyakit blas biasanya menyerang pada semua fase pertumbuhan tanaman padi mulai dari persemaian sampai menjelang panen.',
    
    'blight': 'Penyakit hawar daun pada tanaman padi juga dikenal sebagai bacterial leaf blight (BLB), disebabkan oleh bakteri Xanthomonas oryzae pv. oryzae (Xoo). Gejala penyakit hawar daun dapat bervariasi tergantung pada stadium pertumbuhan tanaman. Selain gejala pada daun, bakteri juga dapat menginfeksi jaringan pembuluh tanaman, menyebabkan gejala yang disebut "kresek". Pada kondisi ini, tanaman padi mengalami layu mendadak, terutama pada fase anakan maksimum hingga pembungaan. Tanaman yang terinfeksi parah akan menghasilkan malai yang tidak terisi atau gabah hampa, yang secara langsung mengurangi hasil panen.',
    
    'Bukan termasuk tanaman padi': 'Bukan termasuk tanaman padi.',
    
    'sehat': 'Tanaman padi dalam kondisi sehat.',
    
    'tungro': 'Penyakit tungro merupakan penyakit padi yang disebabkan oleh dua jenis virus yaitu virus yang berbentuk batang atau virus batang tungro padi Rice tungro bacilliform virus (RTBV), dan virus berbentuk bulat atau virus bulat tungro padi Rice tungro spherical virus (RTSV). Kedua virus tersebut ditularkan oleh beberapa spesies wereng hijau dan wereng daun lainnya. Tanaman padi yang terinfeksi virus-virus tungro umumnya tampak kerdil dan daun berwarna kuning terutama pada daun muda.'
}

# Gejala untuk setiap kelas
class_symptoms = {
    'blast': """
    1. Adanya bercak berbentuk belah ketupat pada daun, pada bagian tepi bercak berwarna kecoklatan dan pada bagian tengah bercak berwarna abu-abu atau putih.
    2. Pada malai terlihat jika ada tangkai malai yang membusuk/busuk leher. Busuk leher menyebabkan gabah hampa jika serangan terjadi sebelum masa pengisian bulir.
    3. Infeksi pada batang menyebabkan busuk batang dan mudah rebah.
    """,
    
    'blight': """
    1. Gejala awal penyakit hawar daun biasanya muncul pada daun muda atau daun yang baru berkembang. 
    2. Gejala serangan ditandai dengan munculnya bercak-bercak kecil berwarna hijau kekuningan pada tepi daun.
    3. Bercak bisa berkembang menjadi garis-garis memanjang berwarna kuning hingga kecoklatan, yang dikenal sebagai "streaking". Garis-garis ini dapat meluas dan menyatu, menyebabkan daun mengering dan mati.
    4. Pada tahap lanjut, daun yang terinfeksi akan tampak seperti terbakar, sehingga penyakit ini disebut "hawar" atau "blight". Pada kondisi parah, hawar dapat meluas ke seluruh daun, menyebabkan daun mengering dan mati.
    """,

    'Bukan termasuk tanaman padi': 'Tidak ada gejala yang relevan.',

    'sehat': 'Tanaman padi dalam kondisi sehat tanpa gejala penyakit.',

    'tungro': """
    1. Gejala utama penyakit tungro terlihat pada perubahan warna daun terutama pada daun muda berwarna kuning oranye dimulai dari ujung daun.
    2. Daun muda agak menggulung, jumlah anakan berkurang, tanaman kerdil dan pertumbuhan terhambat.
    3. Gejala ini biasanya tersebar mengelompok pada areal pertanaman padi sehingga hamparan tanaman padi terlihat bergelombang karena adanya perbedaan tinggi tanaman antara tanaman sehat dan tanaman sakit.
    4. Gejala biasanya mulai tampak pada 6-15 hari setelah terinfeksi. Tanaman muda lebih rentan terinfeksi disbanding tanaman tua.
    """
}

# Tindakan yang harus dilakukan untuk setiap kelas
class_treat = {
    'blast': """
    1. Menggunakan varietas unggul yang tahan terhadap penyakit blas.
    2. Melakukan pemantauan rutin terhadap tanaman padi untuk mendeteksi gejala penyakit sejak dini.
    3. Menanam padi secara serempak, sehingga mudah dalam penanganan.
    4. Menanam dengan jarak yang tidak terlalu rapat, sehingga mengurangi kelembaban.
    5. Tidak menanam padi dengan varietas yang sama secara terus menerus, lakukan rotasi varietas untuk meminimalisir serangan penyakit blas.
    6. Perlakuan benih dengan fungisida, pengendalian dini dengan perlakuan benih sangat dianjurkan untuk menyelamatkan persemaian sampai umur 30 hari setelah sebar.
    7. Penyemprotan dengan fungisida.
    """,
    
    'blight': """
    1. Menggunakan varietas padi yang tahan terhadap penyakit hawar daun (bacterial leaf blight).
    2. Melakukan pemantauan rutin terhadap tanaman padi untuk mendeteksi gejala penyakit sejak dini.
    3. Menghindari genangan air yang berlebihan di lahan sawah.
    4. Membersihkan sisa-sisa tanaman yang terinfeksi dengan cara dibakar.
    5. Alat pertanian juga harus dibersihkan secara rutin dengan desinfektan untuk menghindari kontaminasi.
    6. Meningkatkan ketahanan tanaman dengan pemupukan yang tepat, terutama pupuk makro (N, P, K) dan mikro (Zn, Fe).
    7. Pengendalian gulma secara rutin untuk mengurangi tempat berkembang biak bakteri.
    """,

    'Bukan termasuk tanaman padi': 'Tidak ada tindakan yang diperlukan karena ini bukan tanaman padi.',

    'sehat': 'Tidak ada tindakan yang diperlukan karena tanaman padi dalam kondisi sehat.',

    'tungro': """
    1. Menggunakan varietas padi yang tahan terhadap penyakit tungro.
    2. Melakukan pemantauan rutin terhadap tanaman padi untuk mendeteksi gejala penyakit sejak dini.
    3. Menghindari penanaman padi di lahan yang sebelumnya terinfeksi tungro.
    """
}


# Fungsi prediksi
def predict(image):
    image = image.resize((256, 256)) 
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index]

    return predicted_class, confidence


# === Streamlit UI ===
st.set_page_config(page_title="Klasifikasi Penyakit Tanaman Padi", layout="centered")

st.title("🌾 Klasifikasi Penyakit Tanaman Padi")

st.markdown("---")

uploaded_file = st.file_uploader("Unggah Citra Tanaman Padi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Pratinjau Gambar", use_container_width=True)

    if st.button("🔍 Prediksi"):
        predicted_label, confidence = predict(image)

        st.markdown(f"## ✅ Hasil Prediksi: **{predicted_label.upper()}**")
        st.markdown(f"**Tingkat Keyakinan:** {confidence * 100:.2f}%")
        st.markdown("### 🧠 Deskripsi:")
        st.write(class_description[predicted_label])
        st.markdown("### 🔬 Gejala:")
        st.write(class_symptoms[predicted_label])
        st.markdown("### 🛠️ Tindakan:")
        st.write(class_treat[predicted_label])
