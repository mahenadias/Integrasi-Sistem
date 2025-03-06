import datetime  # UpdateKamis6Maret
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Path untuk menyimpan file bobot
BOBOT_FILE = "bobot_kriteria.json"

MONGO_URI = "mongodb+srv://adilarundaya:Ltp36VZBY3eErO29@adil.rtt73.mongodb.net/?retryWrites=true&w=majority&appName=Adil"
client = MongoClient(MONGO_URI)
db = client["SPK"]
collection_bobot = db["bobot"]
collection_mhs = db["mahasiswa"]
collection_hasil = db["hasil"]
collection_tracking = db["tracking"]


def load_bobot():
    try:
        bobot = collection_bobot.find_one({"_id": "bobot"})
        if bobot:
            return bobot
        else:
            return {
                "waktu_produktif": 30,
                "nilai_tugas": 30,
                "nilai_proyek": 40,
            }
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil bobot: {e}")
        return {
            "waktu_produktif": 30,
            "nilai_tugas": 30,
            "nilai_proyek": 40,
        }


def save_bobot(bobot):
    try:
        collection_bobot.replace_one({"_id": "bobot"}, bobot, upsert=True)
        # st.success("Bobot berhasil disimpan!")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menyimpan bobot: {e}")


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


def load_mahasiswa():
    results = collection_mhs.find({})
    data = []
    for result in results:
        data.append(result)
    return data


def save_mahasiswa(nim, nama, id_meja):
    record = {
        "_id": nim,
        "nama": nama,
        "id": id_meja,
    }
    try:
        collection_mhs.replace_one({"_id": nim}, record, upsert=True)
        st.success(f"Data mahasiswa {nim} berhasil disimpan!")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menyimpan data: {e}")


def edit_mahasiswa(data, nim):
    try:
        collection_mhs.replace_one({"_id": nim}, data, upsert=True)
        st.success(f"Data mahasiswa {nim} berhasil diperbarui!")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menyimpan data: {e}")


def delete_mahasiswa(nim):
    try:
        collection_mhs.delete_one({"_id": nim})
        st.success(f"Data mahasiswa {nim} berhasil dihapus!")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menghapus data: {e}")


def save_data(data, role=None):
    if role is None:
        role = st.session_state.get("role", "unknown_user")
    try:
        collection_hasil.replace_one({"_id": role}, data, upsert=True)
        # st.success("Data berhasil disimpan!")
    except ConnectionFailure as e:
        st.error(f"Terjadi kesalahan saat menyimpan data: {e}")


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
        cctv_url = "https://hack-dealer-whether-remain.trycloudflare.com/"
        st.write(f"Tonton CCTV di [klik di sini]({cctv_url}) jika tidak muncul.")
        st.components.v1.html(
            f"""
            <iframe src="{cctv_url}" width="1920" height="1080" frameborder="0" allowfullscreen></iframe>
        """,
            height=800,
        )
    elif menu == "Input Nilai":
        st.title("Input Nilai Mahasiswa")

        # Membaca daftar mahasiswa dari file JSON
        daftar_mahasiswa = load_mahasiswa()

        # Tanggal otomatis diambil dari datetime dan ditampilkan (tidak bisa diedit)
        tanggal_input = datetime.date.today()
        st.write(f"Tanggal Input: {tanggal_input}")

        # Pilih mahasiswa dari daftar
        nim = st.selectbox(
            "Pilih Mahasiswa",
            options=[""] + [mhs["_id"] for mhs in daftar_mahasiswa],
            format_func=lambda x: (
                f"{x} - {next((mhs['nama'] for mhs in daftar_mahasiswa if mhs['_id'] == x), '')}"
                if x
                else "Pilih NIM"
            ),
        )

        # Deklarasi awal variabel mahasiswa
        mahasiswa = None
        nilai_tugas = None
        nilai_proyek = None
        waktu_produktif = None

        if nim:
            print(f"NIM yang dipilih: {nim}")
            # Ambil data mahasiswa dari daftar
            mahasiswa_data = next(
                (mhs for mhs in daftar_mahasiswa if mhs["_id"] == nim), None
            )
            print(f"Mahasiswa data: {mahasiswa_data}")
            if mahasiswa_data:
                id_mahasiswa = mahasiswa_data["id"]
                nama_mahasiswa = mahasiswa_data["nama"]
            else:
                st.error(f"Data mahasiswa dengan NIM {nim} tidak ditemukan.")
                return

            # Format nama meja berdasarkan ID Mahasiswa
            nama_meja = f"meja{id_mahasiswa}"
            print(f"Nama meja: {nama_meja}")

            # Query data berdasarkan nama meja
            data_mahasiswa = collection_tracking.find_one(
                {nama_meja: {"$exists": True}},
                sort=[
                    ("_id", -1)
                ],  # Sorting berdasarkan _id untuk mendapatkan data terakhir
            )

            if data_mahasiswa:
                # Ambil waktu produktif dari meja yang sesuai
                waktu_produktif = data_mahasiswa[nama_meja]

                # Cek apakah data sudah ada di MongoDB
                existing_data = collection_hasil.find_one({"id": id_mahasiswa})

                if existing_data:
                    nilai_tugas = existing_data.get("nilai_tugas", "belum terisi")
                    nilai_proyek = existing_data.get("nilai_proyek", "belum terisi")
                else:
                    # Siapkan data baru
                    nilai_tugas = "belum terisi"
                    nilai_proyek = "belum terisi"

                    # Insert data baru ke MongoDB
                    new_data = {
                        "id": id_mahasiswa,
                        "nim": nim,
                        "nama": nama_mahasiswa,
                        "waktu_produktif": waktu_produktif,
                        "nilai_tugas": None,
                        "nilai_proyek": None,
                    }
                    collection_hasil.insert_one(new_data)

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
            if mahasiswa is None and nim:
                data_to_save = {
                    "tanggal": str(tanggal_input),
                    "id": id_mahasiswa,
                    "nim": nim,
                    "nama": nama_mahasiswa,
                    "waktu_produktif": waktu_produktif,
                    "nilai_tugas": nilai_tugas,
                    "nilai_proyek": nilai_proyek,
                }

                # Update atau insert ke MongoDB
                collection_hasil.update_one(
                    {"id": id_mahasiswa}, {"$set": data_to_save}, upsert=True
                )
                st.success(f"Nilai untuk mahasiswa {nama_mahasiswa} berhasil disimpan!")

    elif menu == "Hasil SPK":
        st.title("Hasil SPK Mahasiswa")
        # Get results from MongoDB collection_hasil
        results = list(collection_hasil.find({}))

        if not results:
            st.warning("Belum ada data mahasiswa untuk dihitung.")
        else:
            # Ambil bobot dari MongoDB atau default
            bobot_kriteria = collection_bobot.find_one({"_id": "bobot"}) or {
                "waktu_produktif": 30,
                "nilai_tugas": 30,
                "nilai_proyek": 40,
            }

            # Konversi data MongoDB ke format yang dibutuhkan
            data_mahasiswa = []
            for result in results:
                # Skip document if it's the bobot document
                if "_id" in result and result["_id"] == "bobot":
                    continue

                data = {
                    "id": result.get("id"),
                    "waktu_produktif": result.get("waktu_produktif", 0),
                    "nilai_tugas": result.get("nilai_tugas", 0),
                    "nilai_proyek": result.get("nilai_proyek", 0),
                }
                data_mahasiswa.append(data)

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

            # Update hasil SAW ke MongoDB
            if st.button("Simpan Hasil"):
                for hasil in hasil_saw:
                    collection_hasil.update_one(
                        {"id": hasil["id"]}, {"$set": {"hasil_saw": hasil["hasil_saw"]}}
                    )
                st.success("Hasil SPK berhasil disimpan ke MongoDB!")

    elif menu == "Edit Hasil SPK":
        st.title("Edit Hasil SPK")

        # Get all unique dates from hasil collection
        unique_dates = collection_hasil.distinct("tanggal")

        if not unique_dates:
            st.warning("Tidak ada data SPK yang tersedia.")
            return

        # Sort dates in descending order
        tanggal_list = sorted(unique_dates, reverse=True)

        # Pilih tanggal untuk melihat hasil SPK yang sudah ada
        tanggal_pilih = st.selectbox("Pilih Tanggal", options=[""] + tanggal_list)

        if tanggal_pilih:
            # Get data for selected date from MongoDB
            results = list(collection_hasil.find({"tanggal": tanggal_pilih}))

            if not results:
                st.warning("Belum ada data yang disimpan untuk tanggal ini.")
            else:
                # Get list of NIMs
                nim_list = [result.get("nim") for result in results if "nim" in result]
                nim_pilih = st.selectbox(
                    "Pilih NIM yang ingin diedit", options=[""] + nim_list
                )

                if nim_pilih:
                    # Get student data from MongoDB
                    mahasiswa = collection_hasil.find_one(
                        {"nim": nim_pilih, "tanggal": tanggal_pilih}
                    )

                    if mahasiswa:
                        st.subheader(
                            f"Edit Data untuk NIM {nim_pilih} - {mahasiswa.get('nama', 'Nama tidak tersedia')}"
                        )

                        # Display current values
                        st.write(f"Nilai Tugas: {mahasiswa.get('nilai_tugas', 0)}")
                        st.write(f"Nilai Proyek: {mahasiswa.get('nilai_proyek', 0)}")

                        # Form for editing values
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

                        # Save changes button
                        if st.button("Simpan Perubahan"):
                            try:
                                # Update document in MongoDB
                                collection_hasil.update_one(
                                    {"nim": nim_pilih, "tanggal": tanggal_pilih},
                                    {
                                        "$set": {
                                            "nilai_tugas": nilai_tugas,
                                            "nilai_proyek": nilai_proyek,
                                        }
                                    },
                                )
                                st.success(
                                    f"Perubahan berhasil disimpan untuk NIM {nim_pilih}!"
                                )
                            except Exception as e:
                                st.error(f"Gagal menyimpan perubahan: {str(e)}")
                    else:
                        st.error("Data mahasiswa tidak ditemukan.")

    elif menu == "Kelola Mahasiswa":
        st.title("Kelola Data Mahasiswa")

        # Membaca daftar mahasiswa dari file JSON
        daftar_mahasiswa = load_mahasiswa()
        print(daftar_mahasiswa)

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
                        save_mahasiswa(new_nim, new_nama, new_id)
                else:
                    st.error("Lengkapi semua input!")

        elif pilihan == "Edit Mahasiswa":
            st.subheader("Edit Mahasiswa")
            if daftar_mahasiswa:
                nim_edit = st.selectbox(
                    "Pilih NIM untuk diedit",
                    options=[mhs["_id"] for mhs in daftar_mahasiswa],
                )

                if nim_edit:
                    mahasiswa = next(
                        (mhs for mhs in daftar_mahasiswa if mhs["_id"] == nim_edit),
                        None,
                    )
                    if mahasiswa:
                        st.write(f"**NIM:** {nim_edit}")

                        # Input untuk data baru
                        new_nama = st.text_input("Nama Baru", value=mahasiswa["nama"])
                        new_id = st.number_input(
                            "ID (Meja) Baru", value=mahasiswa["id"], min_value=1, step=1
                        )

                        if st.button("Simpan Perubahan"):
                            # Update data mahasiswa
                            mahasiswa["nama"] = new_nama
                            mahasiswa["id"] = int(new_id)
                            edit_mahasiswa(mahasiswa, nim_edit)
                    else:
                        st.error("Mahasiswa tidak ditemukan.")
            else:
                st.warning("Tidak ada data mahasiswa untuk diedit.")

        elif pilihan == "Hapus Mahasiswa":
            st.subheader("Hapus Mahasiswa")
            if daftar_mahasiswa:
                nim_hapus = st.selectbox(
                    "Pilih NIM untuk dihapus",
                    options=[mhs["_id"] for mhs in daftar_mahasiswa],
                )

                if st.button("Hapus Mahasiswa"):
                    print(f"NIM yang dihapus: {nim_hapus}")
                    mahasiswa_dihapus = next(
                        (mhs for mhs in daftar_mahasiswa if mhs["_id"] == nim_hapus),
                        None,
                    )
                    if mahasiswa_dihapus:
                        nama_dihapus = mahasiswa_dihapus["nama"]
                        print(f"Nama yang dihapus: {nama_dihapus}")
                        delete_mahasiswa(nim_hapus)
                    else:
                        st.error("NIM tidak ditemukan!")
            else:
                st.warning("Tidak ada data mahasiswa untuk dihapus.")

        # Menampilkan daftar mahasiswa terkini dengan tabel berwarna
        st.subheader("Daftar Mahasiswa")

        if daftar_mahasiswa:
            # Convert the list of dictionaries to a pandas DataFrame
            df = pd.DataFrame(daftar_mahasiswa)
            df.columns = ["NIM", "Nama", "ID (Meja)"]

            # Display the DataFrame in Streamlit
            st.dataframe(df)
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
    try:

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
    except ConnectionError as e:
        st.error(f"Terjadi kesalahan: {e}")


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

        # Membaca daftar mahasiswa dari file JSON
        daftar_mahasiswa = load_mahasiswa()

        # Tanggal otomatis diambil dari datetime dan ditampilkan (tidak bisa diedit)
        tanggal_input = datetime.date.today()
        st.write(f"Tanggal Input: {tanggal_input}")

        # Pilih mahasiswa dari daftar
        nim = st.selectbox(
            "Pilih Mahasiswa",
            options=[""] + [mhs["_id"] for mhs in daftar_mahasiswa],
            format_func=lambda x: (
                f"{x} - {next((mhs['nama'] for mhs in daftar_mahasiswa if mhs['_id'] == x), '')}"
                if x
                else "Pilih NIM"
            ),
        )

        # Deklarasi awal variabel mahasiswa
        mahasiswa = None
        nilai_tugas = None
        nilai_proyek = None
        waktu_produktif = None

        if nim:
            print(f"NIM yang dipilih: {nim}")
            # Ambil data mahasiswa dari daftar
            mahasiswa_data = next(
                (mhs for mhs in daftar_mahasiswa if mhs["_id"] == nim), None
            )
            print(f"Mahasiswa data: {mahasiswa_data}")
            if mahasiswa_data:
                id_mahasiswa = mahasiswa_data["id"]
                nama_mahasiswa = mahasiswa_data["nama"]
            else:
                st.error(f"Data mahasiswa dengan NIM {nim} tidak ditemukan.")
                return

            # Format nama meja berdasarkan ID Mahasiswa
            nama_meja = f"meja{id_mahasiswa}"
            print(f"Nama meja: {nama_meja}")

            # Query data berdasarkan nama meja
            data_mahasiswa = collection_tracking.find_one(
                {nama_meja: {"$exists": True}},
                sort=[
                    ("_id", -1)
                ],  # Sorting berdasarkan _id untuk mendapatkan data terakhir
            )

            if data_mahasiswa:
                # Ambil waktu produktif dari meja yang sesuai
                waktu_produktif = data_mahasiswa[nama_meja]

                # Cek apakah data sudah ada di MongoDB
                existing_data = collection_hasil.find_one({"id": id_mahasiswa})

                if existing_data:
                    nilai_tugas = existing_data.get("nilai_tugas", "belum terisi")
                    nilai_proyek = existing_data.get("nilai_proyek", "belum terisi")
                else:
                    # Siapkan data baru
                    nilai_tugas = "belum terisi"
                    nilai_proyek = "belum terisi"

                    # Insert data baru ke MongoDB
                    new_data = {
                        "id": id_mahasiswa,
                        "nim": nim,
                        "nama": nama_mahasiswa,
                        "waktu_produktif": waktu_produktif,
                        "nilai_tugas": None,
                        "nilai_proyek": None,
                    }
                    collection_hasil.insert_one(new_data)

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
            if mahasiswa is None and nim:
                data_to_save = {
                    "tanggal": str(tanggal_input),
                    "id": id_mahasiswa,
                    "nim": nim,
                    "nama": nama_mahasiswa,
                    "waktu_produktif": waktu_produktif,
                    "nilai_tugas": nilai_tugas,
                    "nilai_proyek": nilai_proyek,
                }

                # Update atau insert ke MongoDB
                collection_hasil.update_one(
                    {"id": id_mahasiswa}, {"$set": data_to_save}, upsert=True
                )
                st.success(f"Nilai untuk mahasiswa {nama_mahasiswa} berhasil disimpan!")

    elif menu == "Hasil SPK":
        st.title("Hasil SPK Mahasiswa")
        # Get results from MongoDB collection_hasil
        results = list(collection_hasil.find({}))

        if not results:
            st.warning("Belum ada data mahasiswa untuk dihitung.")
        else:
            # Ambil bobot dari MongoDB atau default
            bobot_kriteria = collection_bobot.find_one({"_id": "bobot"}) or {
                "waktu_produktif": 30,
                "nilai_tugas": 30,
                "nilai_proyek": 40,
            }

            # Konversi data MongoDB ke format yang dibutuhkan
            data_mahasiswa = []
            for result in results:
                # Skip document if it's the bobot document
                if "_id" in result and result["_id"] == "bobot":
                    continue

                data = {
                    "id": result.get("id"),
                    "waktu_produktif": result.get("waktu_produktif", 0),
                    "nilai_tugas": result.get("nilai_tugas", 0),
                    "nilai_proyek": result.get("nilai_proyek", 0),
                }
                data_mahasiswa.append(data)

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

            # Update hasil SAW ke MongoDB
            if st.button("Simpan Hasil"):
                for hasil in hasil_saw:
                    collection_hasil.update_one(
                        {"id": hasil["id"]}, {"$set": {"hasil_saw": hasil["hasil_saw"]}}
                    )
                st.success("Hasil SPK berhasil disimpan ke MongoDB!")

    elif menu == "Logout":
        st.session_state.clear()
        st.write("Anda telah logout.")
        st.rerun()


# Menjalankan aplikasi
if __name__ == "__main__":
    try:
        client.admin.command("ismaster")
        print("Connected to MongoDB")

        databases = client.list_database_names()
        print(f"Available databases: {databases}")
    except ConnectionFailure as e:
        print(f"Error connecting to MongoDB: {e}")
    main()
