import os
import glob
import cv2
from vcm.qr_drawing import draw_qr, check_qr_codes

# Input arguments ---------------------------------

# Input/output
src_videos_dir = "/root/data/video_call_mos_set/ref_original"
output_dir = "/root/data/video_call_mos_set/ref_qr"

# QR code options
n_markers = 2 # number of QR code boxes in each frame
dist = 10 # QR code distance to border of video
box_size = 12 # size of QR code
border = 1 # size of QR code border
err = 'h' # QR code error correction

# Quality of output video
crf = 17 # constant rate factor - sets the bitrate for h264
res_output = '1920x1080' # resolution of output video

# Input video requiremnts
res_input = [1920, 1080]

# create videos with QR codes -------------------------------------------
src_videos = glob.glob(os.path.join(src_videos_dir, "*.mp4"))
if len(src_videos)<1:
    raise RuntimeError('No source videos found')

print(f'Starting to draw QR codes for {len(src_videos)} videos')
for src_video in src_videos:
    cap = cv2.VideoCapture(src_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width==res_input[0] and height==res_input[1]):
        marker_pos, output_video = draw_qr(
            src_video,
            output_dir,
            n_markers=n_markers,
            dist=dist,
            box_size=box_size,
            border=border,
            err=err,
            crf=crf,
            res_output=res_output,
            )
        src_file_size = os.path.getsize(src_video) / (1024*1024.0)
        dst_file_size = os.path.getsize(output_video) / (1024*1024.0)
        print(f"Video created. File size: original {src_file_size:0.2f} MB, with qr code {dst_file_size:0.2f} MB")
    else:
        print(f"Skipping video with resolution {width}x{height}: {src_video}")
print(f"marker_pos needed for qr detection of the video call mos model:\n{marker_pos}\n") # marker_pos used during detection for cutting out qr codes 

# check QR codes of created videos  -------------------------------------------
dst_videos = glob.glob(os.path.join(output_dir, "*.mp4"))
print(f'Starting to check QR codes for {len(dst_videos)} videos')
for dst_video in dst_videos:
    check_qr_codes(dst_video)