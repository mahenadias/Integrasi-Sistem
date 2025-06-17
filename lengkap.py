import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from pymongo.mongo_client import MongoClient

# Koneksi ke MongoDB
client = MongoClient(st.secrets["mongo"]["uri"])
db = client["SPK"]
collection_activity = db["tracking"]


# Fungsi konversi waktu
def format_waktu(detik):
    jam = int(detik // 3600)
    sisa = int(detik % 3600)
    menit = sisa // 60
    sisa_detik = sisa % 60

    if jam > 0:
        return f"{jam}h {menit}m {sisa_detik}s"
    else:
        return f"{menit}m {sisa_detik}s"


# Ambil data dari MongoDB
data = list(collection_activity.find())

# Mapping nama mahasiswa
nama_mahasiswa = {
    "meja1": "Alfito Ari Praditya",
    "meja2": "M. Hassanal Mu'izam Hassanudin",
    "meja3": "Fakhri Ahmad Kurnia",
    "meja4": "Rizqy Achmad Fahreza",
    "meja5": "Dimas Tri Atmojo",
    "meja6": "Muhammad Fawwaz Fauzan",
}

# Nim Mahasiswa
nim_mahasiswa = {
    "meja1": "5312422015",
    "meja2": "5312422042",
    "meja3": "5312422016",
    "meja4": "5312422031",
    "meja5": "5312422033",
    "meja6": "5312422032",
}

# Ambil data duduk dan tidur
if data:
    data = data[0]

    duduk = {
        "meja1": data.get("meja1", 0),
        "meja2": data.get("meja2", 0),
        "meja3": data.get("meja3", 0),
        "meja4": data.get("meja4", 0),
        "meja5": data.get("meja5", 0),
        "meja6": data.get("meja6", 0),
    }

    tidur = {
        "meja1": data.get("sleepmeja1", 0),
        "meja2": data.get("sleepmeja2", 0),
        "meja3": data.get("sleepmeja3", 0),
        "meja4": data.get("sleepmeja4", 0),
        "meja5": data.get("sleepmeja5", 0),
        "meja6": data.get("sleepmeja6", 0),
    }

    # Max dan min untuk SAW
    max_duduk = max(duduk.values())
    min_tidur = min(tidur.values())

    # Bobot
    bobot_duduk = 0.7
    bobot_tidur = 0.3

    # Hitung skor
    hasil = []
    for meja in duduk.keys():
        nilai_duduk = duduk[meja] / max_duduk
        nilai_tidur = 1 if tidur[meja] == 0 else min_tidur / tidur[meja]
        skor = (nilai_duduk * bobot_duduk) + (nilai_tidur * bobot_tidur)

        # Konversi waktu detik ke string format
        duduk_waktu = format_waktu(duduk[meja])
        tidur_waktu = format_waktu(tidur[meja])

        hasil.append(
            {
                "Nama": nama_mahasiswa[meja],
                "NIM": nim_mahasiswa[meja],
                "Duduk (waktu)": duduk_waktu,
                "Tidur (waktu)": tidur_waktu,
                "Skor Produktivitas": round(skor, 4),
            }
        )

    # Buat DataFrame
    df = pd.DataFrame(hasil)
    df = df.sort_values(by="Skor Produktivitas", ascending=False)

    # Tampilkan tabel tanpa index
    st.title("Hasil Produktivitas Mahasiswa")
    st.table(df.to_dict(orient="records"))

    # Visualisasi chart
    fig, ax = plt.subplots()
    ax.bar(df["Nama"], df["Skor Produktivitas"], color="cornflowerblue")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Skor Produktivitas")
    plt.title("Ranking Produktivitas Mahasiswa")
    st.pyplot(fig)

else:
    st.write("Belum ada data aktivitas yang tersimpan di database.")
