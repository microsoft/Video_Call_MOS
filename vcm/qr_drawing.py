import os
import shutil
import subprocess
import numpy as np
import qrcode
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
import cv2

def draw_qr(
    src_video,
    output_dir,
    n_markers=2,
    dist=10,
    box_size=12,
    border=1,
    err='h',
    crf=15,
    res_output='1920x1080',
    ):

    if err=='l':
        error_correction = qrcode.constants.ERROR_CORRECT_L
    elif err=='m':
        error_correction = qrcode.constants.ERROR_CORRECT_M
    elif err=='h':
        error_correction = qrcode.constants.ERROR_CORRECT_H

    cap = cv2.VideoCapture(src_video)

    # Get input video parameters:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_codec = int(cap.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Input video: {} \nSize: {} x {}, FPS: {}, Video codec: {}, Frame count: {}'.format(src_video, width, height, fps, video_codec, total_frames))

    # Dump images
    print("Dumping images ...")
    tmp_dir = os.path.join(output_dir, 'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    src_tmp_dir = os.path.join(tmp_dir, 'src')
    os.makedirs(src_tmp_dir, exist_ok=True)

    dst_tmp_dir = os.path.join(tmp_dir, 'dst')
    os.makedirs(dst_tmp_dir, exist_ok=True)

    img_template = 'img_%05d.bmp'
    img_template_path = os.path.join(src_tmp_dir, img_template)

    ff_args = [
        'ffmpeg',
        '-i',
        src_video,
        img_template_path
    ]
    run_process(ff_args)

    # Add QR code to dumped images
    print("Adding QR codes to dumped images ...")
    for frame in range(1, total_frames+1):
        img_path = img_template_path % frame
        img_marked_path = os.path.join(dst_tmp_dir, os.path.basename(img_path))
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        qr = qrcode.QRCode(
            version=None,
            error_correction=error_correction,
            box_size=box_size,
            border=border,
        )
        qr.add_data(frame)
        qr.make(fit=True)

        img_qr = qr.make_image(fill_color="black", back_color="white")
        img_qr = np.array(img_qr.convert('RGB'))[:, :, ::-1].copy()

        if n_markers==1:
            markers = [[1080-dist-img_qr.shape[0],1920-dist-img_qr.shape[1]]]
        elif n_markers==2:
            markers = [[dist,dist], [1080-dist-img_qr.shape[0],1920-dist-img_qr.shape[1]]]
        elif n_markers==3:
            markers = [[dist,dist], [1080-dist-img_qr.shape[0],1920-dist-img_qr.shape[1]], [1080-dist-img_qr.shape[0], dist]]
        elif n_markers==4:
            markers = [[dist,dist], [1080-dist-img_qr.shape[0],1920-dist-img_qr.shape[1]], [dist,1920-dist-img_qr.shape[1]], [1080-dist-img_qr.shape[0], dist]]

        for marker in markers:
            img[marker[0]:marker[0]+img_qr.shape[0], marker[1]:marker[1]+img_qr.shape[1]] = img_qr

        # Write img with qr code
        cv2.imwrite(img_marked_path, img)

    # Encode new video file based on marked images and copy audio from source video file
    img_marked_template = os.path.join(dst_tmp_dir, img_template)
    output_video = os.path.join(output_dir, os.path.splitext(os.path.basename(src_video))[0] + '_qr.mp4')

    ff_args = [
        'ffmpeg',
        '-y',
        '-framerate',
        f'{fps}',
        '-i',
        img_marked_template,
        '-i',
        src_video,
        '-c:a copy',
        '-vcodec',
        'libx264',
        '-s',
        res_output,
        '-crf',
        f'{crf}',
        '-preset',
        'slow',
        '-pix_fmt',
        'yuv420p',
        '-map 0:v:0',
        '-map 1:a:0',
        output_video,
    ]
    print("Encoding new video based on images with QR codes ...")
    run_process(ff_args)
    shutil.rmtree(tmp_dir)

    marker_pos = {
        'marker_1_x1': markers[0][1],
        'marker_1_x2': markers[0][1]+img_qr.shape[0], 
        'marker_1_y1': markers[0][0],
        'marker_1_y2': markers[0][0]+img_qr.shape[1], 
        'marker_2_x1': markers[1][1], 
        'marker_2_x2': markers[1][1]+img_qr.shape[0], 
        'marker_2_y1': markers[1][0],
        'marker_2_y2': markers[1][0]+img_qr.shape[1], 
        }

    return marker_pos, output_video

def run_process(cmd_args):
    cmd_args = ' '.join(cmd_args)
    status = subprocess.run(cmd_args, check=False, shell=True, capture_output=True)
    if status.returncode != 0:
        raise RuntimeError(status.stderr)
    return status.returncode
    
def check_qr_codes(dst_video):
    print(f'Input video: {dst_video}')
    cap_deg = cv2.VideoCapture(dst_video)
    n_frames = int(cap_deg.get(cv2. CAP_PROP_FRAME_COUNT))

    for frame in range(1, n_frames+1):
        _ , img = cap_deg.read()
        barcodes = pyzbar.decode(img, symbols=[ZBarSymbol.QRCODE])
        if len(barcodes)<1:
            raise RuntimeError(f"No QR code found for frame: {frame}")
        for barcode in barcodes:
            detected_frame = int(barcode.data.decode("utf-8"))
            if detected_frame!=frame:
                raise RuntimeError(f"Wrong frame number for QR code found. Actual frame: {frame}, detected frame {detected_frame}")
    print(f'--> All QR codes succesfully detected')