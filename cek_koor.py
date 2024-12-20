import cv2


# Fungsi callback untuk mouse
def get_coordinates(event, x, y, flags, param):
    global x1, y1, x2, y2, clicked
    if event == cv2.EVENT_LBUTTONDOWN:  # Ketika tombol kiri mouse ditekan
        if not clicked:
            x1, y1 = x, y  # Titik awal (x1, y1)
            clicked = True
        else:
            x2, y2 = x, y  # Titik akhir (x2, y2)
            clicked = False
            print(f"Koordinat meja: ({x1}, {y1}), ({x2}, {y2})")


# Baca gambar atau frame video
frame = cv2.imread("scene00151.png")  # Atau gunakan VideoCapture untuk video

# # Ubah ukuran gambar menjadi 384x640
# new_width = 640
# new_height = 384
# frame = cv2.resize(frame, (new_width, new_height))

clicked = False
x1, y1, x2, y2 = 0, 0, 0, 0

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", get_coordinates)

while True:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):  # Tekan 'q' untuk keluar
        break

cv2.destroyAllWindows()
