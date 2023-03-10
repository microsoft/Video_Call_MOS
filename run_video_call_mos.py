import argparse
from vcm.model import VideoCallMosModel

parser = argparse.ArgumentParser()
parser.add_argument("--deg_video", type=str)
parser.add_argument("--ref_video", type=str)
parser.add_argument("--results_dir", type=str)
parser.add_argument("--tmp_dir", type=str)
args = parser.parse_args()

vcm_model = VideoCallMosModel(checkpoint="vcm/video_call_mos_weights.pt")

mos, results_csv = vcm_model(args.deg_video, args.ref_video, args.results_dir, args.tmp_dir, verbosity=1)
print(f"MOS: {mos:0.2f}, deg_video {args.deg_video}, ref_video {args.ref_video}, results csv: {results_csv}")