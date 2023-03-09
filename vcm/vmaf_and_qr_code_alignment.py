import os
import cv2
import subprocess
import shutil
import glob
import pandas as pd
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol

vidqr_marker_pos={'marker_1_x1': 10, 'marker_1_x2': 286, 'marker_1_y1': 10, 'marker_1_y2': 286, 'marker_2_x1': 1634, 'marker_2_x2': 1910, 'marker_2_y1': 794, 'marker_2_y2': 1070}
vidqr_buffer=-10
vidqr_border_size=10

def get_vmaf(deg_video, ref_video, tmp_dir, clean_up=True, verbosity=1):
    vmaf_results_csv = prepare(deg_video, ref_video, tmp_dir)
    if verbosity:
        print("Detecting QR codes of degraded video")
    ref_frames = detect_frames(deg_video, verbosity=verbosity)
    if verbosity:
        print("Dumping frames of reference videos as images")
    ref_images_template = dump_reference_images(ref_video, tmp_dir, verbosity=verbosity)
    if verbosity:
        print("Aligning dumped reference frames to degraded video")    
    align_reference_images(ref_frames, tmp_dir, ref_images_template)
    if verbosity:
        print("Creating aligned reference video")        
    create_aligned_reference_video(tmp_dir, aligned_ref='aligned_ref.yuv', verbosity=verbosity)
    if verbosity:
        print("Computing VMAF with aligned reference video")        
    run_vmaf(deg_video, vmaf_results_csv, tmp_dir, aligned_ref='aligned_ref.yuv', verbosity=verbosity)
    df_vmaf = pd.read_csv(vmaf_results_csv)
    df_vmaf['ref_frames'] = ref_frames
    if clean_up:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    if verbosity:
        print("VMAF computation done")
    return df_vmaf

def prepare(deg_video, ref_video, tmp_dir):
    if not os.path.exists(ref_video):
        raise ValueError(f"Ref video not found! Path: {ref_video}")
    if not os.path.exists(deg_video):
        raise ValueError(f"Deg video not found! Path: {deg_video}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir)    
    sub_dirs = ["ref_img_dir", "aligned_ref_img_dir"]
    for sub_dir in sub_dirs:
        tmp_sub_dir = os.path.join(tmp_dir, sub_dir)
        shutil.rmtree(tmp_sub_dir, ignore_errors=True)
        os.makedirs(tmp_sub_dir)
    vmaf_results_csv = os.path.join(tmp_dir, os.path.basename(deg_video).split('.')[0]+'.csv')
    return vmaf_results_csv

def detect_single_frame(img, marker_pos, buffer, border_size):
    img_1 = img[marker_pos['marker_1_y1']-buffer:marker_pos['marker_1_y2']+buffer, marker_pos['marker_1_x1']-buffer:marker_pos['marker_1_x2']+buffer]
    img_2 = img[marker_pos['marker_2_y1']-buffer:marker_pos['marker_2_y2']+buffer, marker_pos['marker_2_x1']-buffer:marker_pos['marker_2_x2']+buffer]
    img_1 = cv2.copyMakeBorder(img_1, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,value=[255,255,255])
    img_2 = cv2.copyMakeBorder(img_2, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT,value=[255,255,255])
    barcodes = pyzbar.decode(img_1, symbols=[ZBarSymbol.QRCODE]) + pyzbar.decode(img_2, symbols=[ZBarSymbol.QRCODE])                            
    detected_frames = []
    for barcode in barcodes:
        data = barcode.data.decode("utf-8")
        if data.isdigit():
            detected_frames.append(int(data))
    detected_frame = None
    if len(detected_frames)>0:
        if all(x==detected_frames[0] for x in detected_frames):
            detected_frame = detected_frames[0]
        else:
            detected_frame = max(detected_frames)
    return detected_frame

def detect_frames(
    deg_video,
    verbosity=1,
    ):
    cap_deg = cv2.VideoCapture(deg_video)
    frames_deg = int(cap_deg.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_frames = []
    for i in range(frames_deg):
        _ , img = cap_deg.read()
        detected_frame = detect_single_frame(img, vidqr_marker_pos, vidqr_buffer, vidqr_border_size)
        if detected_frame is None:
            buffer = -30
            while detected_frame is None and buffer<=vidqr_marker_pos['marker_1_x1']:
                print(f"{i+1}  -- QR Code not detected. Retrying with detection window buffer size = {buffer}")
                detected_frame = detect_single_frame(img, vidqr_marker_pos, buffer, vidqr_border_size)
                buffer = buffer+1
        ref_frames.append(detected_frame)
    not_detected = sum(x is None for x in ref_frames)
    if verbosity:
        print(f"QR Detection done. ref_frames length: {len(ref_frames)}. not_detected: {not_detected}")
    if (len(ref_frames)==0) or (not_detected!=0):
        raise RuntimeError("Not all frames detected")
    return ref_frames

def dump_reference_images(ref_video, tmp_dir, img_template='orig_img_%05d.bmp', verbosity=1):
    first_frame_number = detect_single_frame(cv2.VideoCapture(ref_video).read()[1], vidqr_marker_pos, vidqr_buffer, vidqr_border_size)    
    ref_images_template = os.path.join(tmp_dir, "ref_img_dir", img_template)
    ff_args = [
        'ffmpeg',
        '-i',
        ref_video,
        "-start_number",
        f"{first_frame_number}",
        ref_images_template,
    ]
    run_process(ff_args, verbosity)
    return ref_images_template

def align_reference_images(ref_frames, tmp_dir, ref_images_template, img_template='aligned_img_%05d.bmp'):
    for cnt, ref_frame in enumerate(ref_frames):
        ref_img_path = ref_images_template % ref_frame
        aligned_ref_image = img_template % (cnt+1)
        aligned_ref_image_path = os.path.join(tmp_dir, "aligned_ref_img_dir", aligned_ref_image)
        shutil.copyfile(ref_img_path, aligned_ref_image_path)
    img_cnt = len(glob.glob(os.path.join(tmp_dir, "aligned_ref_img_dir", f"*.{img_template.split('.')[-1]}")))
    if img_cnt != len(ref_frames):
        raise RuntimeError(f"Error: Created {img_cnt} of {len(ref_frames)} expected images.")

def create_aligned_reference_video(tmp_dir, img_template='aligned_img_%05d.bmp', aligned_ref='aligned_ref.yuv', verbosity=1):
    aligned_ref_img_template = os.path.join(tmp_dir, "aligned_ref_img_dir", img_template)
    aligned_ref_vid_path = os.path.join(tmp_dir, aligned_ref)
    ff_args = [
        'ffmpeg',
        '-y',
        '-i',
        aligned_ref_img_template,
        '-pix_fmt',
        'yuv420p',
        '-f',
        'rawvideo',
        aligned_ref_vid_path,
    ]
    run_process(ff_args, verbosity)

def run_vmaf(deg_video, vmaf_results, tmp_dir, deg_fps=30, aligned_ref='aligned_ref.yuv', verbosity=1):
    aligned_ref_vid_path = os.path.join(tmp_dir, aligned_ref)
    ff_args = [
        'ffmpeg',
        '-i',
        deg_video,        
        '-video_size',
        '1920x1080',
        '-framerate',
        f"{deg_fps}",
        '-pixel_format',
        'yuv420p',
        '-i',
        aligned_ref_vid_path,
        '-lavfi',
        f"libvmaf='feature=name=psnr|name=float_ssim:log_fmt=csv:log_path={vmaf_results}:n_threads={os.cpu_count()}'",
        '-f',
        'null',
        '-',
    ]
    run_process(ff_args, verbosity)

def run_process(cmd_args, verbosity=1):
    cmd_args = ' '.join(cmd_args)
    if verbosity:
        print("Running command '{}'".format(cmd_args))
    status = subprocess.run(cmd_args, check=False, shell=True, capture_output=True)
    if status.returncode != 0:
        raise RuntimeError(status.stderr)
    return status.returncode