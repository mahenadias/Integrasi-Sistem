import argparse
import csv
import json  # Add this to the imports
import os
import platform
import sys
import time
from pathlib import Path

from pymongo import MongoClient

# Koneksi ke MongoDB
url = "mongodb+srv://adilarundaya:Ltp36VZBY3eErO29@adil.rtt73.mongodb.net/?retryWrites=true&w=majority&appName=Adil"
client = MongoClient(
    url
)  # Ganti dengan URL MongoDB Anda jika menggunakan server berbeda
db = client["SPK"]  # Nama database
collection_face = db["presensi"]  # Collection untuk data face recognition
collection_activity = db["tracking"]  # Collection untuk data aktivitas


import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import (
    IMG_FORMATS,
    VID_FORMATS,
    LoadImages,
    LoadScreenshots,
    LoadStreams,
)
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(1980, 1080),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):

    source = str(source)
    save_img = not nosave and not source.endswith(".txt")
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = (
        source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    )
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(
            source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride
        )
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(
            source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride
        )
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Timer and table areas
    total_fps = 0
    frame_count = 0
    meja1_time = 0
    meja2_time = 0
    meja3_time = 0
    start_time_meja1 = None
    start_time_meja2 = None
    start_time_meja3 = None
    meja1_x1, meja1_y1 = 407, 366
    meja1_x2, meja1_y2 = 407, 366
    meja2_x1, meja2_y1 = 346, 105
    meja2_x2, meja2_y2 = 346, 105
    meja3_x1, meja3_y1 = 268, 105
    meja3_x2, meja3_y2 = 268, 105
    meja4_x1, meja4_y1 = 167, 117
    meja4_x2, meja4_y2 = 167, 117
    prev_time = time.time()
    last_count_time = time.time()
    last_save_time = time.time()
    timer = time.time()
    counts_meja1 = 0
    counts_meja2 = 0
    counts_meja3 = 0
    counts_meja4 = 0
    save_interval = 5  # seconds

    # Helper to check if person is within a table area
    def is_within_table(x, y, table_x1, table_y1, table_x2, table_y2):
        return table_x1 <= x <= table_x2 and table_y1 <= y <= table_y2

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = (
        0,
        [],
        (Profile(device=device), Profile(device=device), Profile(device=device)),
    )

    def generate_grid_points(x_min, y_min, x_max, y_max):
        width = x_max - x_min
        height = y_max - y_min
        grid_points = []
        for i in range(3):
            for j in range(3):
                point_x = x_min + width * i / 2
                point_y = y_min + height * j / 2
                grid_points.append((point_x, point_y))
        return grid_points

        # Buat titik grid di meja

    def generate_table_points(table_x1, table_y1, table_x2, table_y2, grid_size=3):
        width = table_x2 - table_x1
        height = table_y2 - table_y1
        points = []
        for i in range(grid_size):
            for j in range(grid_size):
                px = table_x1 + i * width / (grid_size - 1)
                py = table_y1 + j * height / (grid_size - 1)
                points.append((px, py))
        return points

    def is_point_in_box(px, py, box_x_min, box_y_min, box_x_max, box_y_max):
        return box_x_min <= px <= box_x_max and box_y_min <= py <= box_y_max

    for path, im, im0s, vid_cap, s in dataset:
        start_time = time.time()
        current_time = time.time()
        is_in_table1 = is_in_table2 = is_in_table3 = is_in_table4 = False
        # Inisialisasi teks meja dengan "No" di awal setiap frame
        meja1_text = "No"
        meja2_text = "No"
        meja3_text = "No"
        meja4_text = "No"

        # Hitung FPS
        end_time = time.time()  # Catat waktu selesai pemrosesan
        elapsed_time = end_time - prev_time  # Waktu antara frame
        prev_time = end_time
        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        total_fps += fps
        frame_count += 1

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = (
                increment_path(save_dir / Path(path).stem, mkdir=True)
                if visualize
                else False
            )
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(
                            image, augment=augment, visualize=visualize
                        ).unsqueeze(0)
                    else:
                        pred = torch.cat(
                            (
                                pred,
                                model(
                                    image, augment=augment, visualize=visualize
                                ).unsqueeze(0),
                            ),
                            dim=0,
                        )
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
            )

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {
                "Image Name": image_name,
                "Prediction": prediction,
                "Confidence": confidence,
            }
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Function to save data to JSON file
        def save_results_to_json():
            results = {
                "total_time_meja1": counts_meja1,
                "total_time_meja2": counts_meja2,
                "total_time_meja3": counts_meja3,
            }
            json_path = save_dir / "meja_times.json"
            with open(json_path, "w") as json_file:
                json.dump(results, json_file, indent=4)
            print(f"Results autosaved to {json_path}")

        # Process predictions
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )
            s += "{:g}x{:g} ".format(*im.shape[2:])
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                table1_points = generate_table_points(
                    meja1_x1, meja1_y1, meja1_x2, meja1_y2
                )
                table2_points = generate_table_points(
                    meja2_x1, meja2_y1, meja2_x2, meja2_y2
                )
                table3_points = generate_table_points(
                    meja3_x1, meja3_y1, meja3_x2, meja3_y2
                )
                table4_points = generate_table_points(
                    meja4_x1, meja4_y1, meja4_x2, meja4_y2
                )

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:
                        if save_format == 0:
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                                .view(-1)
                                .tolist()
                            )
                        else:
                            coords = (
                                (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                            )
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    # Detect if the person is within the table areas

                    if label == "person":
                        grid_points = generate_grid_points(
                            xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        )
                        # # Draw grid lines
                        # for i in range(1, 3):  # 2 vertical and horizontal lines
                        #     # Vertical lines
                        #     start_x = int(xyxy[0] + (xyxy[2] - xyxy[0]) * i / 3)
                        #     end_x = start_x
                        #     start_y = int(xyxy[1])
                        #     end_y = int(xyxy[3])
                        #     cv2.line(
                        #         im0, (start_x, start_y), (end_x, end_y), (255, 0, 0), 1
                        #     )

                        #     # Horizontal lines
                        #     start_x = int(xyxy[0])
                        #     end_x = int(xyxy[2])
                        #     start_y = int(xyxy[1] + (xyxy[3] - xyxy[1]) * i / 3)
                        #     end_y = start_y
                        #     cv2.line(
                        #         im0, (start_x, start_y), (end_x, end_y), (255, 0, 0), 1
                        #     )

                        # Check each point in the grid for each table
                        for px, py in table1_points:
                            if is_point_in_box(
                                px, py, xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                            ):
                                is_in_table1 = True

                        for px, py in table2_points:
                            if is_point_in_box(
                                px, py, xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                            ):
                                is_in_table2 = True

                        for px, py in table3_points:
                            if is_point_in_box(
                                px, py, xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                            ):
                                is_in_table3 = True

                        for px, py in table4_points:
                            if is_point_in_box(
                                px, py, xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                            ):
                                is_in_table4 = True

                    if save_img or save_crop or view_img:
                        label = (
                            None
                            if hide_labels
                            else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        )
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(
                            xyxy,
                            imc,
                            file=save_dir / "crops" / names[c] / f"{p.stem}.jpg",
                            BGR=True,
                        )

                # Display text for each table area
                meja1_text = "Yes" if is_in_table1 else "No"
                meja2_text = "Yes" if is_in_table2 else "No"
                meja3_text = "Yes" if is_in_table3 else "No"
                meja4_text = "Yes" if is_in_table4 else "No"

            # Perbarui counts setiap 1 detik
            if current_time - last_count_time >= 1:
                if is_in_table1:
                    counts_meja1 += (
                        1  # Tambahkan ke hitungan Meja 1 jika terdeteksi "Yes"
                    )
                if is_in_table2:
                    counts_meja2 += (
                        1  # Tambahkan ke hitungan Meja 2 jika terdeteksi "Yes"
                    )
                if is_in_table3:
                    counts_meja3 += (
                        1  # Tambahkan ke hitungan Meja 3 jika terdeteksi "Yes"
                    )
                if is_in_table4:
                    counts_meja4 += (
                        1  # Tambahkan ke hitungan Meja 3 jika terdeteksi "Yes"
                    )
                last_count_time = current_time  # Update the last save time
                # Autosave logic
                if current_time - last_save_time >= 10:
                    # Data JSON dari sistem Tracking Aktivitas (contoh)
                    results = {
                        "meja1": counts_meja1,
                        "meja2": counts_meja2,
                        "meja3": counts_meja3,
                        "meja4": counts_meja4,
                    }
                    # Menyimpan data aktivitas ke MongoDB
                    collection_activity.insert_one(results)
                    # save_results_to_json()
                    last_save_time = current_time  # Update the last save time

            cv2.rectangle(
                im0, (meja1_x1, meja1_y1), (meja1_x2, meja1_y2), (0, 255, 0), 2
            )
            cv2.rectangle(
                im0, (meja2_x1, meja2_y1), (meja2_x2, meja2_y2), (0, 255, 0), 2
            )
            cv2.rectangle(
                im0, (meja3_x1, meja3_y1), (meja3_x2, meja3_y2), (0, 255, 0), 2
            )
            cv2.rectangle(
                im0, (meja4_x1, meja4_y1), (meja4_x2, meja4_y2), (0, 255, 0), 2
            )
            cv2.putText(
                im0,
                f"{meja1_text}: {counts_meja1}",
                (meja1_x1, meja1_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                im0,
                f"{meja2_text}: {counts_meja2}",
                (meja2_x1, meja2_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                im0,
                f"{meja3_text}: {counts_meja3}",
                (meja3_x1, meja3_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                im0,
                f"{meja4_text}: {counts_meja4}",
                (meja4_x1, meja4_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )
            # Menampilkan jumlah deteksi "Yes" per frame
            cv2.putText(
                im0,
                f"Save time - {current_time - last_count_time:.1f} Last Save - {current_time - last_save_time:.1f} Timer - {current_time - timer:.1f} Meja1: {counts_meja1} | Meja2: {counts_meja2} | Meja3: {counts_meja3}",
                (10, 30),  # Posisi pada frame, sesuaikan sesuai kebutuhan
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Warna: Merah
                2,
            )
            cv2.putText(
                im0,
                f"FPS: {fps:.2f}",
                (10, 50),  # Posisi teks di frame
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # Ukuran font
                (0, 255, 0),  # Warna teks (hijau)
                2,  # Ketebalan teks
            )

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(
                        str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
                    )  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            average_fps = total_fps / frame_count if frame_count > 0 else 0
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = average_fps, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                    vid_writer.write(im0)

        # Print time (inference-only)
        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms"
        )

    # Print final accumulated times
    print(f"Waktu total di Meja 1: {counts_meja1} detik")
    print(f"Waktu total di Meja 2: {counts_meja2} detik")
    print(f"Waktu total di Meja 3: {counts_meja3} detik")
    print(f"Waktu total di Meja 4: {counts_meja4} detik")

    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}"
        % t
    )
    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
            if save_txt
            else ""
        )
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])


def parse_opt():
    """
    Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        --weights (str | list[str], optional): Model path or Triton URL. Defaults to ROOT / 'yolov5s.pt'.
        --source (str, optional): File/dir/URL/glob/screen/0(webcam). Defaults to ROOT / 'data/images'.
        --data (str, optional): Dataset YAML path. Provides dataset configuration information.
        --imgsz (list[int], optional): Inference size (height, width). Defaults to [640].
        --conf-thres (float, optional): Confidence threshold. Defaults to 0.25.
        --iou-thres (float, optional): NMS IoU threshold. Defaults to 0.45.
        --max-det (int, optional): Maximum number of detections per image. Defaults to 1000.
        --device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        --view-img (bool, optional): Flag to display results. Defaults to False.
        --save-txt (bool, optional): Flag to save results to *.txt files. Defaults to False.
        --save-csv (bool, optional): Flag to save results in CSV format. Defaults to False.
        --save-conf (bool, optional): Flag to save confidences in labels saved via --save-txt. Defaults to False.
        --save-crop (bool, optional): Flag to save cropped prediction boxes. Defaults to False.
        --nosave (bool, optional): Flag to prevent saving images/videos. Defaults to False.
        --classes (list[int], optional): List of classes to filter results by, e.g., '--classes 0 2 3'. Defaults to None.
        --agnostic-nms (bool, optional): Flag for class-agnostic NMS. Defaults to False.
        --augment (bool, optional): Flag for augmented inference. Defaults to False.
        --visualize (bool, optional): Flag for visualizing features. Defaults to False.
        --update (bool, optional): Flag to update all models in the model directory. Defaults to False.
        --project (str, optional): Directory to save results. Defaults to ROOT / 'runs/detect'.
        --name (str, optional): Sub-directory name for saving results within --project. Defaults to 'exp'.
        --exist-ok (bool, optional): Flag to allow overwriting if the project/name already exists. Defaults to False.
        --line-thickness (int, optional): Thickness (in pixels) of bounding boxes. Defaults to 3.
        --hide-labels (bool, optional): Flag to hide labels in the output. Defaults to False.
        --hide-conf (bool, optional): Flag to hide confidences in the output. Defaults to False.
        --half (bool, optional): Flag to use FP16 half-precision inference. Defaults to False.
        --dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Defaults to False.
        --vid-stride (int, optional): Video frame-rate stride, determining the number of frames to skip in between
            consecutive frames. Defaults to 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Example:
        ```python
        from ultralytics import YOLOv5
        args = YOLOv5.parse_opt()
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolov5s.pt",
        help="model path or triton URL",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "data/images",
        help="file/dir/URL/glob/screen/0(webcam)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=ROOT / "data/coco128.yaml",
        help="(optional) dataset.yaml path",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.25, help="confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold"
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximum detections per image"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument(
        "--save-csv", action="store_true", help="save results in CSV format"
    )
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--save-crop", action="store_true", help="save cropped prediction boxes"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default=ROOT / "runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    parser.add_argument(
        "--line-thickness", default=1, type=int, help="bounding box thickness (pixels)"
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels"
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences"
    )
    parser.add_argument(
        "--half", action="store_true", help="use FP16 half-precision inference"
    )
    parser.add_argument(
        "--dnn", action="store_true", help="use OpenCV DNN for ONNX inference"
    )
    parser.add_argument(
        "--vid-stride", type=int, default=1, help="video frame-rate stride"
    )
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 model inference based on provided command-line arguments, validating dependencies before running.

    Args:
        opt (argparse.Namespace): Command-line arguments for YOLOv5 detection. See function `parse_opt` for details.

    Returns:
        None

    Note:
        This function performs essential pre-execution checks and initiates the YOLOv5 detection process based on user-specified
        options. Refer to the usage guide and examples for more information about different sources and formats at:
        https://github.com/ultralytics/ultralytics

    Example usage:

    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
    ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
