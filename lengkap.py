import datetime
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient

# Path untuk menyimpan file bobot
BOBOT_FILE = "bobot_kriteria.json"

MONGO_URI = "mongodb+srv://adilarundaya:Ltp36VZBY3eErO29@adil.rtt73.mongodb.net/?retryWrites=true&w=majority&appName=Adil"
client = MongoClient(MONGO_URI)


def connect_to_mongodb():
    client = MongoClient(MONGO_URI)
    db = client["SPK"]  # Ganti dengan nama database Anda
    collection = db["tracking"]  # Ganti dengan nama koleksi Anda
    return collection


def get_file_name():
    # Pastikan role sudah ada di session state
    role = st.session_state.get(
        "role", "unknown_role"
    )  # Default jika role tidak ditemukan
    today = datetime.date.today()
    return f"data_mahasiswa_{today}_{role}.json"


def load_bobot():
    if os.path.exists(BOBOT_FILE):
        with open(BOBOT_FILE, "r") as file:
            return json.load(file)
    else:
        return {
            "waktu_produktif": 30,
            "nilai_tugas": 30,
            "nilai_proyek": 40,
        }


def save_bobot(bobot):
    with open(BOBOT_FILE, "w") as file:
        json.dump(bobot, file)


# def ambil_data_dari_mongodb(start_time=None, end_time=None):
#     try:
#         collection = connect_to_mongodb()
#         query = {}
#         if start_time and end_time:
#             query = {"tanggal": {"$gte": start_time, "$lte": end_time}}
#         data = list(
#             collection.find(query, {"_id": 0})
#         )  # Exclude '_id' jika tidak dibutuhkan
#         return data
#     except Exception as e:
#         st.error(f"Terjadi kesalahan saat mengambil data: {e}")
#         return []


def konversi_ke_bobot(nilai, kategori):
    if kategori == "produktif":
        if nilai <= 3:
            return 1
        elif nilai <= 7:
            return 2
        elif nilai <= 11:
            return 3
        elif nilai <= 14:
            return 4
        else:
            return 5
    elif kategori == "nilai_tugas":
        if nilai < 60:
            return 1
        elif nilai <= 69:
            return 2
        elif nilai <= 79:
            return 3
        elif nilai <= 89:
            return 4
        else:
            return 5
    elif kategori == "nilai_proyek":
        if nilai < 60:
            return 1
        elif nilai <= 69:
            return 2
        elif nilai <= 79:
            return 3
        elif nilai <= 89:
            return 4
        else:
            return 5
    return 0


# Fungsi untuk menyimpan hasil ke file
def save_results(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


# Fungsi untuk memuat data dari file berdasarkan tanggal
def load_results(tanggal=None):
    if not tanggal:
        tanggal = datetime.date.today().strftime("%Y-%m-%d")

    filename = f"data_mahasiswa_{tanggal}_admin.json"

    if os.path.exists(filename):
        with open(filename, "r") as file:
            try:
                data = json.load(file)

                # Pastikan data adalah dictionary dan tidak kosong
                if isinstance(data, dict) and data:
                    return data
                else:
                    st.error("Data dalam file tidak valid. Harus berupa dictionary.")
                    return {}
            except json.JSONDecodeError as e:
                st.error(f"Error membaca file JSON: {e}")
                return {}
    else:
        st.warning(f"File {filename} tidak ditemukan.")
        return {}


def load_mahasiswa(filename="daftar_mahasiswa.json"):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


# Fungsi untuk menyimpan daftar mahasiswa ke file JSON
def save_mahasiswa(data, filename="daftar_mahasiswa.json"):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)


def save_data(data, role=None):
    if role is None:
        role = st.session_state.get("role", "unknown_user")
    file_name = get_file_name()
    with open(file_name, "w") as file:
        json.dump(data, file)
    st.success(f"Data berhasil disimpan di {file_name}")


def hitung_saw_dengan_bobot(data, bobot):
    hasil = []
    for index, row in data.iterrows():
        waktu_produktif_bobot = konversi_ke_bobot(row["waktu_produktif"], "produktif")
        nilai_tugas_bobot = konversi_ke_bobot(row["nilai_tugas"], "nilai_tugas")
        nilai_proyek_bobot = konversi_ke_bobot(row["nilai_proyek"], "nilai_proyek")

        st.write(
            f"Row {index}: waktu_produktif={waktu_produktif_bobot}, nilai_tugas={nilai_tugas_bobot}, nilai_proyek={nilai_proyek_bobot}"
        )

        hasil_saw = (
            waktu_produktif_bobot * bobot["waktu_produktif"] / 100
            + nilai_tugas_bobot * bobot["nilai_tugas"] / 100
            + nilai_proyek_bobot * bobot["nilai_proyek"] / 100
        )
        hasil.append({"id": row["id"], "hasil_saw": round(hasil_saw, 2)})
    return hasil


# Fungsi untuk halaman admin
def admin_page():
    st.sidebar.title("Menu Admin")
    menu = st.sidebar.radio(
        "Navigasi",
        [
            "Dashboard",
            "Live CCTV",
            "Input Nilai",
            "Hasil SPK",
            "Edit Hasil SPK",
            "Kelola Mahasiswa",
            "Pengaturan",
            "Logout",
        ],
    )

    if menu == "Dashboard":
        st.title("Dashboard Admin")
        st.write("Selamat datang di halaman admin.")
    elif menu == "Live CCTV":
        st.title("Live CCTV")
        cctv_url = "https://lit-elements-head-bosnia.trycloudflare.com"
        st.write(f"Tonton CCTV di [klik di sini]({cctv_url}) jika tidak muncul.")
        st.components.v1.html(
            f"""
            <iframe src="{cctv_url}" width="1920" height="1080" frameborder="0" allowfullscreen></iframe>
        """,
            height=800,
        )
    elif menu == "Input Nilai":
        st.title("Input Nilai Mahasiswa")
        db = client["SPK"]
        collection = db["tracking"]

        # Membaca daftar mahasiswa dari file JSON
        daftar_mahasiswa = load_mahasiswa()

        # Tanggal otomatis diambil dari datetime dan ditampilkan (tidak bisa diedit)
        tanggal_input = datetime.date.today()
        st.write(f"Tanggal Input: {tanggal_input}")

        # Pilih mahasiswa dari daftar
        nim = st.selectbox(
            "Pilih Mahasiswa",
            options=[""] + list(daftar_mahasiswa.keys()),
            format_func=lambda x: (
                f"{x} - {daftar_mahasiswa[x]['nama']}"
                if x in daftar_mahasiswa
                else "Pilih NIM"
            ),
        )

        # Deklarasi awal variabel mahasiswa
        mahasiswa = None
        nilai_tugas = None
        nilai_proyek = None
        waktu_produktif = None

        # Pastikan session_state memiliki atribut 'data_mahasiswa'
        if "data_mahasiswa" not in st.session_state:
            st.session_state.data_mahasiswa = {}

        if nim:
            # Ambil data mahasiswa dari daftar
            mahasiswa_data = daftar_mahasiswa[nim]
            id_mahasiswa = mahasiswa_data["id"]
            nama_mahasiswa = mahasiswa_data["nama"]

            # Format nama meja berdasarkan ID Mahasiswa
            nama_meja = f"meja{id_mahasiswa}"

            # Query data berdasarkan nama meja
            data_mahasiswa = collection.find_one(
            {nama_meja: {"$exists": True}},
                sort=[
                    ("_id", -1)
                ],  # Sorting berdasarkan _id untuk mendapatkan data terakhir
            )

            if data_mahasiswa:
                # Ambil waktu produktif dari meja yang sesuai
                waktu_produktif = data_mahasiswa[nama_meja]

                # Cek apakah data sudah ada di session state
                if (
                    "data_mahasiswa" in st.session_state
                    and id_mahasiswa in st.session_state.data_mahasiswa
                ):
                    mahasiswa = st.session_state.data_mahasiswa[id_mahasiswa]
                    nilai_tugas = mahasiswa.get("nilai_tugas", "belum terisi")
                    nilai_proyek = mahasiswa.get("nilai_proyek", "belum terisi")
                else:
                    # Simpan data mahasiswa untuk digunakan nanti
                    mahasiswa = {
                        "id": id_mahasiswa,
                        "nim": nim,
                        "nama": nama_mahasiswa,
                        "waktu_produktif": waktu_produktif,
                        "nilai_tugas": None,
                        "nilai_proyek": None,
                    }
                    st.session_state.data_mahasiswa[id_mahasiswa] = mahasiswa

                # Tampilkan informasi mahasiswa
                st.write(f"Waktu Produktif ({nama_meja}): {waktu_produktif} menit")
                st.write(f"Nilai Tugas: {nilai_tugas}")
                st.write(f"Nilai Proyek: {nilai_proyek}")
            else:
                st.write(f"Tidak ditemukan data untuk ID Mahasiswa {id_mahasiswa}.")

        # Tampilkan input nilai dengan nilai default jika tersedia
        nilai_tugas = st.number_input(
            "Nilai Tugas",
            min_value=0,
            max_value=100,
            step=1,
            value=nilai_tugas if isinstance(nilai_tugas, int) else 0,
        )
        nilai_proyek = st.number_input(
            "Nilai Proyek",
            min_value=0,
            max_value=100,
            step=1,
            value=nilai_proyek if isinstance(nilai_proyek, int) else 0,
        )

        if st.button("Simpan Nilai"):
            if "data_mahasiswa" not in st.session_state:
                st.session_state.data_mahasiswa = {}

            if mahasiswa is None:
                mahasiswa = {
                    "tanggal": str(tanggal_input),
                    "id": id_mahasiswa,
                    "nim": nim,
                    "nama": nama_mahasiswa,
                    "waktu_produktif": waktu_produktif,
                    "nilai_tugas": nilai_tugas,
                    "nilai_proyek": nilai_proyek,
                }
                st.session_state.data_mahasiswa[id_mahasiswa] = mahasiswa
            else:
                mahasiswa["nilai_tugas"] = nilai_tugas
                mahasiswa["nilai_proyek"] = nilai_proyek
                st.session_state.data_mahasiswa[id_mahasiswa] = mahasiswa

            # Simpan data ke session state
            st.session_state.data_mahasiswa[id_mahasiswa] = mahasiswa

            # Berikan notifikasi
            st.success(f"Nilai untuk {id_mahasiswa} berhasil disimpan!")
            save_data(
                st.session_state.data_mahasiswa,
                st.session_state.get("role", "unknown_user"),
            )

    elif menu == "Hasil SPK":
        st.title("Hasil SPK Mahasiswa")
        if "data_mahasiswa" in st.session_state:
            data_mahasiswa = list(st.session_state.data_mahasiswa.values())
        else:
            data_mahasiswa = []

        if not data_mahasiswa:
            st.warning("Belum ada data mahasiswa untuk dihitung.")
        else:
            # Ambil bobot dari session_state atau default
            bobot_kriteria = st.session_state.get(
                "bobot_kriteria",
                {
                    "waktu_produktif": 30,
                    "nilai_tugas": 30,
                    "nilai_proyek": 40,
                },
            )

            # Fungsi untuk menghitung SAW dengan bobot dinamis
            def hitung_saw_dengan_bobot(data, bobot):
                hasil = []
                for row in data:
                    waktu_produktif = row["waktu_produktif"] or 0
                    nilai_tugas = row["nilai_tugas"] or 0
                    nilai_proyek = row["nilai_proyek"] or 0

                    hasil_saw = (
                        waktu_produktif * bobot["waktu_produktif"] / 100
                        + nilai_tugas * bobot["nilai_tugas"] / 100
                        + nilai_proyek * bobot["nilai_proyek"] / 100
                    )
                    hasil.append({"id": row["id"], "hasil_saw": hasil_saw})
                return hasil

            hasil_saw = hitung_saw_dengan_bobot(data_mahasiswa, bobot_kriteria)
            hasil_saw = sorted(hasil_saw, key=lambda x: x["hasil_saw"], reverse=False)
            df = pd.DataFrame(hasil_saw)
            df = df.sort_values(by="hasil_saw", ascending=True)
            df["rank"] = range(1, len(df) + 1)

            fig = px.bar(
                df,
                x="hasil_saw",
                y="rank",
                orientation="h",
                title="Peringkat Mahasiswa Berdasarkan Hasil SPK",
                text="id",
            )

            fig.update_traces(
                texttemplate="%{text}",
                textposition="inside",
                marker=dict(color="royalblue"),
            )

            fig.add_scatter(
                x=df["hasil_saw"],
                y=df["rank"],
                mode="text",  # Mode teks
                text=df["hasil_saw"].apply(lambda x: f"{x:.2f}"),  # Format hasil SAW
                textposition="middle right",  # Menempatkan teks di sebelah kanan bar
                showlegend=False,
            )

            fig.update_layout(
                xaxis_title="Hasil SPK (%)",
                yaxis_title="Rank",
                yaxis=dict(
                    tickmode="array",
                    tickvals=df["rank"],
                    ticktext=df["rank"].sort_values(ascending=False).astype(str),
                ),
                showlegend=False,
            )

            st.plotly_chart(fig)

            # Tombol untuk menyimpan hasil
            if st.button("Simpan Hasil"):
                save_results(hasil_saw)
                st.success("Hasil SPK berhasil disimpan ke file `hasil_saw.json`!")

    elif menu == "Edit Hasil SPK":
        st.title("Edit Hasil SPK")

        # Mendapatkan daftar tanggal yang ada di file
        file_names = [
            f
            for f in os.listdir()
            if f.startswith("data_mahasiswa_") and f.endswith("_admin.json")
        ]
        tanggal_list = sorted([f.split("_")[2] for f in file_names])

        if not tanggal_list:
            st.warning("Tidak ada data SPK yang tersedia.")
            return

        # Pilih tanggal untuk melihat hasil SPK yang sudah ada
        tanggal_pilih = st.selectbox("Pilih Tanggal", options=[""] + tanggal_list)

        if tanggal_pilih:
            # Muat data berdasarkan tanggal yang dipilih
            data = load_results(tanggal=tanggal_pilih)

            if not data:
                st.warning("Belum ada data yang disimpan untuk tanggal ini.")
            else:
                # Menampilkan pilihan untuk memilih NIM yang akan diedit
                nim_list = [item["nim"] for item in data.values()]
                nim_pilih = st.selectbox(
                    "Pilih NIM yang ingin diedit", options=[""] + nim_list
                )

                if nim_pilih:
                    # Mencari data berdasarkan NIM
                    mahasiswa = next(
                        (item for item in data.values() if item["nim"] == nim_pilih),
                        None,
                    )

                    if mahasiswa:
                        st.subheader(
                            f"Edit Data untuk NIM {nim_pilih} - {mahasiswa['nama']}"
                        )

                        # Tampilkan data yang ada
                        st.write(f"Nilai Tugas: {mahasiswa['nilai_tugas']}")
                        st.write(f"Nilai Proyek: {mahasiswa['nilai_proyek']}")

                        # Form untuk mengedit nilai
                        nilai_tugas = st.number_input(
                            "Nilai Tugas",
                            min_value=0,
                            max_value=100,
                            value=mahasiswa.get("nilai_tugas", 0),
                        )
                        nilai_proyek = st.number_input(
                            "Nilai Proyek",
                            min_value=0,
                            max_value=100,
                            value=mahasiswa.get("nilai_proyek", 0),
                        )

                        # Update data mahasiswa dengan nilai baru
                        mahasiswa["nilai_tugas"] = nilai_tugas
                        mahasiswa["nilai_proyek"] = nilai_proyek

                        # Tombol untuk menyimpan perubahan
                        if st.button("Simpan Perubahan"):
                            # Simpan perubahan ke file
                            filename = f"data_mahasiswa_{tanggal_pilih}_admin.json"
                            save_results(data, filename)
                            st.success(
                                f"Perubahan berhasil disimpan untuk NIM {nim_pilih}!"
                            )
                    else:
                        st.error("Data mahasiswa tidak ditemukan.")

    elif menu == "Kelola Mahasiswa":
        st.title("Kelola Data Mahasiswa")

        # Membaca daftar mahasiswa dari file JSON
        daftar_mahasiswa = load_mahasiswa()

        # Pilihan tindakan: Tambah atau Hapus Mahasiswa
        pilihan = st.radio(
            "Pilih Tindakan", ["Tambah Mahasiswa", "Edit Mahasiswa", "Hapus Mahasiswa"]
        )

        if pilihan == "Tambah Mahasiswa":
            st.subheader("Tambah Mahasiswa")
            new_nim = st.text_input("NIM")
            new_nama = st.text_input("Nama")
            new_id = st.number_input("ID (Meja)", min_value=1, step=1)

            if st.button("Tambah Mahasiswa"):
                if new_nim and new_nama and new_id:
                    if new_nim in daftar_mahasiswa:
                        st.warning(f"Mahasiswa dengan NIM {new_nim} sudah ada.")
                    else:
                        daftar_mahasiswa[new_nim] = {
                            "id": int(new_id),
                            "nama": new_nama,
                        }
                        save_mahasiswa(daftar_mahasiswa)
                        st.success(f"Mahasiswa {new_nama} berhasil ditambahkan.")
                else:
                    st.error("Lengkapi semua input!")

        elif pilihan == "Edit Mahasiswa":
            st.subheader("Edit Mahasiswa")
            if daftar_mahasiswa:
                nim_edit = st.selectbox(
                    "Pilih NIM untuk diedit", options=list(daftar_mahasiswa.keys())
                )

                if nim_edit:
                    mahasiswa = daftar_mahasiswa[nim_edit]
                    st.write(f"**NIM:** {nim_edit}")

                    # Input untuk data baru
                    new_nama = st.text_input("Nama Baru", value=mahasiswa["nama"])
                    new_id = st.number_input(
                        "ID (Meja) Baru", value=mahasiswa["id"], min_value=1, step=1
                    )

                    if st.button("Simpan Perubahan"):
                        # Update data mahasiswa
                        daftar_mahasiswa[nim_edit]["nama"] = new_nama
                        daftar_mahasiswa[nim_edit]["id"] = int(new_id)
                        save_mahasiswa(daftar_mahasiswa)
                        st.success(f"Data mahasiswa {nim_edit} berhasil diperbarui!")
            else:
                st.warning("Tidak ada data mahasiswa untuk diedit.")

        elif pilihan == "Hapus Mahasiswa":
            st.subheader("Hapus Mahasiswa")
            if daftar_mahasiswa:
                nim_hapus = st.selectbox(
                    "Pilih NIM untuk dihapus", options=list(daftar_mahasiswa.keys())
                )

                if st.button("Hapus Mahasiswa"):
                    if nim_hapus in daftar_mahasiswa:
                        nama_dihapus = daftar_mahasiswa[nim_hapus]["nama"]
                        del daftar_mahasiswa[nim_hapus]
                        save_mahasiswa(daftar_mahasiswa)
                        st.success(
                            f"Mahasiswa {nama_dihapus} dengan NIM {nim_hapus} berhasil dihapus."
                        )
                    else:
                        st.error("NIM tidak ditemukan!")
            else:
                st.warning("Tidak ada data mahasiswa untuk dihapus.")

        # Menampilkan daftar mahasiswa terkini dengan tabel berwarna
        st.subheader("Daftar Mahasiswa")

        if daftar_mahasiswa:
            # Konversi data ke DataFrame
            df_mahasiswa = pd.DataFrame.from_dict(daftar_mahasiswa, orient="index")
            df_mahasiswa.reset_index(inplace=True)
            df_mahasiswa.columns = ["NIM", "ID", "Nama"]

            # Styling tabel
            def highlight_row(row):
                return [
                    (
                        "background-color: #f4f4f4"
                        if row.name % 2 == 0
                        else "background-color: #eaf3ff"
                    )
                ] * len(row)

            styled_table = df_mahasiswa.style.apply(highlight_row, axis=1)

            # Menampilkan tabel dengan gaya
            st.dataframe(styled_table, use_container_width=True)
        else:
            st.info("Belum ada data mahasiswa.")

    elif menu == "Pengaturan":
        st.title("Pengaturan")
        if "bobot_kriteria" not in st.session_state:
            st.session_state.bobot_kriteria = load_bobot()

        produktif_bobot = st.slider(
            "Bobot Waktu Produktif (%)",
            0,
            100,
            st.session_state.bobot_kriteria["waktu_produktif"],
        )
        tugas_bobot = st.slider(
            "Bobot Nilai Tugas (%)",
            0,
            100,
            st.session_state.bobot_kriteria["nilai_tugas"],
        )
        proyek_bobot = st.slider(
            "Bobot Nilai Proyek (%)",
            0,
            100,
            st.session_state.bobot_kriteria["nilai_proyek"],
        )

        total_bobot = produktif_bobot + tugas_bobot + proyek_bobot
        if total_bobot != 100:
            st.warning(f"Total bobot harus 100%. Total saat ini: {total_bobot}%")
        else:
            if st.button("Simpan Bobot"):
                st.session_state.bobot_kriteria = {
                    "waktu_produktif": produktif_bobot,
                    "nilai_tugas": tugas_bobot,
                    "nilai_proyek": proyek_bobot,
                }
                save_bobot(st.session_state.bobot_kriteria)
                st.success("Bobot kriteria berhasil disimpan!")

        st.write(f"Bobot saat ini: {st.session_state.bobot_kriteria}")

    elif menu == "Logout":
        st.session_state.clear()
        st.write("Anda telah logout.")
        st.rerun()


# Halaman utama
def main():
    st.set_page_config(
        page_title="Sistem Pengambilan Keputusan Mahasiswa", layout="wide"
    )
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.session_state.role = "admin"
                st.success("Login berhasil!")
                admin_page()
            elif username == "dosen" and password == "dosen123":
                st.session_state.logged_in = True
                st.session_state.role = "dosen"
                st.success("Login berhasil!")
                dosen_page()
            else:
                st.error("Username atau password salah!")
    else:
        if st.session_state.role == "admin":
            admin_page()
        elif st.session_state.role == "dosen":
            dosen_page()


# Halaman dosen
def dosen_page():
    st.sidebar.title("Menu Dosen")
    menu = st.sidebar.radio(
        "Navigasi", ["Dashboard", "Input Nilai", "Hasil SPK", "Logout"]
    )

    if menu == "Dashboard":
        st.title("Dashboard Dosen")
        st.write("Selamat datang di halaman dosen.")
    elif menu == "Input Nilai":
        st.title("Input Nilai Mahasiswa")
        db = client["SPK"]
        collection = db["tracking"]

        # Membaca daftar mahasiswa dari file JSON
        daftar_mahasiswa = load_mahasiswa()

        # Tanggal otomatis diambil dari datetime dan ditampilkan (tidak bisa diedit)
        tanggal_input = datetime.date.today()
        st.write(f"Tanggal Input: {tanggal_input}")

        # Pilih mahasiswa dari daftar
        nim = st.selectbox(
            "Pilih Mahasiswa",
            options=[""] + list(daftar_mahasiswa.keys()),
            format_func=lambda x: (
                f"{x} - {daftar_mahasiswa[x]['nama']}"
                if x in daftar_mahasiswa
                else "Pilih NIM"
            ),
        )

        # Deklarasi awal variabel mahasiswa
        mahasiswa = None

        if nim:
            # Ambil data mahasiswa dari daftar
            mahasiswa_data = daftar_mahasiswa[nim]
            id_mahasiswa = mahasiswa_data["id"]
            nama_mahasiswa = mahasiswa_data["nama"]

            # Format nama meja berdasarkan ID Mahasiswa
            nama_meja = f"meja{id_mahasiswa}"

            # Query data berdasarkan nama meja
            data_mahasiswa = collection.find_one({nama_meja: {"$exists": True}})

            if data_mahasiswa:
                # Ambil waktu produktif dari meja yang sesuai
                waktu_produktif = data_mahasiswa[nama_meja]

                # Simpan data mahasiswa untuk digunakan nanti
                mahasiswa = {
                    "tanggal": str(tanggal_input),
                    "id": id_mahasiswa,
                    "nim": nim,
                    "nama": nama_mahasiswa,
                    "waktu_produktif": waktu_produktif,
                    "nilai_tugas": None,
                    "nilai_proyek": None,
                }

                # Tampilkan data mahasiswa
                st.write(f"Tanggal: {tanggal_input}")
                st.write(f"ID Mahasiswa: {id_mahasiswa}")
                st.write(f"Waktu Produktif ({nama_meja}): {waktu_produktif} menit")
            else:
                st.write(f"Tidak ditemukan data untuk ID Mahasiswa {id_mahasiswa}.")

        nilai_tugas = st.number_input("Nilai Tugas", min_value=0, max_value=100, step=1)
        nilai_proyek = st.number_input(
            "Nilai Proyek", min_value=0, max_value=100, step=1
        )

        if st.button("Simpan Nilai"):
            if mahasiswa is None:
                mahasiswa = {
                    "tanggal": str(tanggal_input),
                    "id": id_mahasiswa,
                    "nim": nim,
                    "nama": nama_mahasiswa,
                    "waktu_produktif": waktu_produktif,
                    "nilai_tugas": None,
                    "nilai_proyek": None,
                }
                st.session_state.data_mahasiswa[id_mahasiswa] = mahasiswa
            else:
                mahasiswa["nilai_tugas"] = nilai_tugas
                mahasiswa["nilai_proyek"] = nilai_proyek
                st.session_state.data_mahasiswa[id_mahasiswa] = mahasiswa
            st.success(f"Nilai untuk {id_mahasiswa} berhasil disimpan!")
            save_data(
                st.session_state.data_mahasiswa,
                st.session_state.get("role", "unknown_user"),
            )
    elif menu == "Hasil SPK":
        st.title("Hasil SPK Mahasiswa")
        if "data_mahasiswa" in st.session_state:
            data_mahasiswa = list(st.session_state.data_mahasiswa.values())
        else:
            data_mahasiswa = []

        if not data_mahasiswa:
            st.warning("Belum ada data mahasiswa untuk dihitung.")
        else:
            # Ambil bobot dari session_state atau default
            bobot_kriteria = st.session_state.get(
                "bobot_kriteria",
                {
                    "waktu_produktif": 30,
                    "nilai_tugas": 30,
                    "nilai_proyek": 40,
                },
            )

            # Fungsi untuk menghitung SAW dengan bobot dinamis
            def hitung_saw_dengan_bobot(data, bobot):
                hasil = []
                for row in data:
                    waktu_produktif = row["waktu_produktif"] or 0
                    nilai_tugas = row["nilai_tugas"] or 0
                    nilai_proyek = row["nilai_proyek"] or 0

                    hasil_saw = (
                        waktu_produktif * bobot["waktu_produktif"] / 100
                        + nilai_tugas * bobot["nilai_tugas"] / 100
                        + nilai_proyek * bobot["nilai_proyek"] / 100
                    )
                    hasil.append({"id": row["id"], "hasil_saw": hasil_saw})
                return hasil

            hasil_saw = hitung_saw_dengan_bobot(data_mahasiswa, bobot_kriteria)
            hasil_saw = sorted(hasil_saw, key=lambda x: x["hasil_saw"], reverse=False)
            df = pd.DataFrame(hasil_saw)
            df = df.sort_values(by="hasil_saw", ascending=True)
            df["rank"] = range(1, len(df) + 1)

            fig = px.bar(
                df,
                x="hasil_saw",
                y="rank",
                orientation="h",
                title="Peringkat Mahasiswa Berdasarkan Hasil SPK",
                text="id",
            )

            fig.update_traces(
                texttemplate="%{text}",
                textposition="inside",
                marker=dict(color="royalblue"),
            )

            fig.add_scatter(
                x=df["hasil_saw"],
                y=df["rank"],
                mode="text",  # Mode teks
                text=df["hasil_saw"].apply(lambda x: f"{x:.2f}"),  # Format hasil SAW
                textposition="middle right",  # Menempatkan teks di sebelah kanan bar
                showlegend=False,
            )

            fig.update_layout(
                xaxis_title="Hasil SPK (%)",
                yaxis_title="Rank",
                yaxis=dict(
                    tickmode="array",
                    tickvals=df["rank"],
                    ticktext=df["rank"].sort_values(ascending=False).astype(str),
                ),
                showlegend=False,
            )

            st.plotly_chart(fig)

            # Tombol untuk menyimpan hasil
            if st.button("Simpan Hasil"):
                save_results(hasil_saw)
                st.success("Hasil SPK berhasil disimpan ke file `hasil_saw.json`!")

    elif menu == "Logout":
        st.session_state.clear()
        st.write("Anda telah logout.")
        st.rerun()


# Menjalankan aplikasi
if __name__ == "__main__":
    main()
