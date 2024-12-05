import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

# Path untuk menyimpan file bobot
BOBOT_FILE = "bobot_kriteria.json"


# Fungsi untuk memuat bobot dari file
def load_bobot():
    if os.path.exists(BOBOT_FILE):
        with open(BOBOT_FILE, "r") as file:
            return json.load(file)
    else:
        # Jika file tidak ada, kembalikan bobot default
        return {
            "kehadiran": 15,
            "waktu_produktif": 15,
            "nilai_tugas": 30,
            "nilai_proyek": 40,
        }


# Fungsi untuk menyimpan bobot ke file
def save_bobot(bobot):
    with open(BOBOT_FILE, "w") as file:
        json.dump(bobot, file)


# Dummy function untuk MongoDB dan SAW
def ambil_data_dari_mongodb(start_time, end_time):
    return [
        {
            "id": "01",
            "kehadiran": 80,
            "waktu_produktif": 60,
            "nilai_tugas": 0,  # Nilai tugas awal kosong
            "nilai_proyek": 0,  # Nilai proyek awal kosong
        },
        {
            "id": "02",
            "kehadiran": 70,
            "waktu_produktif": 50,
            "nilai_tugas": 0,  # Nilai tugas awal kosong
            "nilai_proyek": 0,  # Nilai proyek awal kosong
        },
    ]


# Fungsi untuk menghitung SAW
def hitung_saw(data):
    BOBOT = {
        "kehadiran": 0.15,
        "waktu_produktif": 0.15,
        "nilai_tugas": 0.3,
        "nilai_proyek": 0.4,
    }
    hasil = []
    for row in data:
        kehadiran = row["kehadiran"] if row["kehadiran"] is not None else 0
        waktu_produktif = (
            row["waktu_produktif"] if row["waktu_produktif"] is not None else 0
        )
        nilai_tugas = row["nilai_tugas"] if row["nilai_tugas"] is not None else 0
        nilai_proyek = row["nilai_proyek"] if row["nilai_proyek"] is not None else 0

        hasil_saw = (
            kehadiran * BOBOT["kehadiran"]
            + waktu_produktif * BOBOT["waktu_produktif"]
            + nilai_tugas * BOBOT["nilai_tugas"]
            + nilai_proyek * BOBOT["nilai_proyek"]
        )
        hasil.append({"id": row["id"], "hasil_saw": hasil_saw})
    return hasil


# Fungsi untuk halaman admin
def admin_page():
    st.sidebar.title("Menu Admin")
    menu = st.sidebar.radio(
        "Navigasi",
        ["Dashboard", "Live CCTV", "Input Nilai", "Hasil SPK", "Pengaturan", "Logout"],
    )

    if menu == "Dashboard":
        st.title("Dashboard Admin")
        st.write("Selamat datang di halaman admin.")
    elif menu == "Live CCTV":
        st.title("Live CCTV")
        cctv_url = "http://192.168.18.224:8081/"
        st.write(f"Tonton CCTV di [klik di sini]({cctv_url}) jika tidak muncul.")
        st.components.v1.html(
            f"""
            <iframe src="{cctv_url}" width="800" height="600" frameborder="0" allowfullscreen></iframe>
        """,
            height=600,
        )
    elif menu == "Input Nilai":
        st.title("Input Nilai Mahasiswa")
        # Input ID Mahasiswa
        id_mahasiswa = st.text_input("Masukkan ID Mahasiswa")
        if "data_mahasiswa" not in st.session_state:
            st.session_state.data_mahasiswa = {}
        if id_mahasiswa in st.session_state.data_mahasiswa:
            mahasiswa = st.session_state.data_mahasiswa[id_mahasiswa]
            st.write(f"ID Mahasiswa: {mahasiswa['id']}")
            st.write(f"Kehadiran: {mahasiswa['kehadiran']}%")
            st.write(f"Waktu Produktif: {mahasiswa['waktu_produktif']} menit")
        else:
            mahasiswa = None
            st.write(
                f"ID Mahasiswa {id_mahasiswa} belum ditemukan, silakan masukkan nilai."
            )

        nilai_tugas = st.number_input("Nilai Tugas", min_value=0, max_value=100, step=1)
        nilai_proyek = st.number_input(
            "Nilai Proyek", min_value=0, max_value=100, step=1
        )

        if st.button("Simpan Nilai"):
            if mahasiswa is None:
                mahasiswa = {
                    "id": id_mahasiswa,
                    "kehadiran": None,
                    "waktu_produktif": None,
                    "nilai_tugas": nilai_tugas,
                    "nilai_proyek": nilai_proyek,
                }
                st.session_state.data_mahasiswa[id_mahasiswa] = mahasiswa
            else:
                mahasiswa["nilai_tugas"] = nilai_tugas
                mahasiswa["nilai_proyek"] = nilai_proyek
                st.session_state.data_mahasiswa[id_mahasiswa] = mahasiswa
            st.success(f"Nilai untuk {id_mahasiswa} berhasil disimpan!")

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
                    "kehadiran": 15,
                    "waktu_produktif": 15,
                    "nilai_tugas": 30,
                    "nilai_proyek": 40,
                },
            )

            # Fungsi untuk menghitung SAW dengan bobot dinamis
            def hitung_saw_dengan_bobot(data, bobot):
                hasil = []
                for row in data:
                    kehadiran = row["kehadiran"] or 0
                    waktu_produktif = row["waktu_produktif"] or 0
                    nilai_tugas = row["nilai_tugas"] or 0
                    nilai_proyek = row["nilai_proyek"] or 0

                    hasil_saw = (
                        kehadiran * bobot["kehadiran"] / 100
                        + waktu_produktif * bobot["waktu_produktif"] / 100
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

    elif menu == "Pengaturan":
        st.title("Pengaturan")
        if "bobot_kriteria" not in st.session_state:
            st.session_state.bobot_kriteria = load_bobot()

        kehadiran_bobot = st.slider(
            "Bobot Kehadiran (%)", 0, 100, st.session_state.bobot_kriteria["kehadiran"]
        )
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

        total_bobot = kehadiran_bobot + produktif_bobot + tugas_bobot + proyek_bobot
        if total_bobot != 100:
            st.warning(f"Total bobot harus 100%. Total saat ini: {total_bobot}%")
        else:
            if st.button("Simpan Bobot"):
                st.session_state.bobot_kriteria = {
                    "kehadiran": kehadiran_bobot,
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
        id_mahasiswa = st.text_input("Masukkan ID Mahasiswa")
        if "data_mahasiswa" not in st.session_state:
            st.session_state.data_mahasiswa = {}
        if id_mahasiswa in st.session_state.data_mahasiswa:
            mahasiswa = st.session_state.data_mahasiswa[id_mahasiswa]
            st.write(f"ID Mahasiswa: {mahasiswa['id']}")
            st.write(f"Kehadiran: {mahasiswa['kehadiran']}%")
            st.write(f"Waktu Produktif: {mahasiswa['waktu_produktif']} menit")
        else:
            mahasiswa = None
            st.write(
                f"ID Mahasiswa {id_mahasiswa} belum ditemukan, silakan masukkan nilai."
            )

        nilai_tugas = st.number_input("Nilai Tugas", min_value=0, max_value=100, step=1)
        nilai_proyek = st.number_input(
            "Nilai Proyek", min_value=0, max_value=100, step=1
        )

        if st.button("Simpan Nilai"):
            if mahasiswa is None:
                mahasiswa = {
                    "id": id_mahasiswa,
                    "kehadiran": None,
                    "waktu_produktif": None,
                    "nilai_tugas": nilai_tugas,
                    "nilai_proyek": nilai_proyek,
                }
                st.session_state.data_mahasiswa[id_mahasiswa] = mahasiswa
            else:
                mahasiswa["nilai_tugas"] = nilai_tugas
                mahasiswa["nilai_proyek"] = nilai_proyek
                st.session_state.data_mahasiswa[id_mahasiswa] = mahasiswa
            st.success(f"Nilai untuk {id_mahasiswa} berhasil disimpan!")

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
                    "kehadiran": 15,
                    "waktu_produktif": 15,
                    "nilai_tugas": 30,
                    "nilai_proyek": 40,
                },
            )

            # Fungsi untuk menghitung SAW dengan bobot dinamis
            def hitung_saw_dengan_bobot(data, bobot):
                hasil = []
                for row in data:
                    kehadiran = row["kehadiran"] or 0
                    waktu_produktif = row["waktu_produktif"] or 0
                    nilai_tugas = row["nilai_tugas"] or 0
                    nilai_proyek = row["nilai_proyek"] or 0

                    hasil_saw = (
                        kehadiran * bobot["kehadiran"] / 100
                        + waktu_produktif * bobot["waktu_produktif"] / 100
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

            fig.add_scatter(
                x=df["hasil_saw"],
                y=["rank"],
                mode="text",
                text=df["hasil_saw"].apply(lambda x: f"{x:.2f}"),
                textposition="middle right",
                showlegend=False,
                textfont=dict(weight="bold"),
            )

            fig.update_traces(
                texttemplate="%{text}",
                # textposition="inside",
                marker=dict(color="royalblue"),
            )

            fig.update_layout(
                xaxis_title="Hasil SPK (%)",
                yaxis_title=None,
                yaxis=dict(
                    tickmode="array",
                    tickvals=df["rank"],
                    ticktext=df["rank"].astype(str),
                    categoryorder="total ascending",
                ),
                showlegend=False,
            )

            st.plotly_chart(fig)

    elif menu == "Logout":
        st.session_state.clear()
        st.write("Anda telah logout.")
        st.rerun()


# Menjalankan aplikasi
if __name__ == "__main__":
    main()