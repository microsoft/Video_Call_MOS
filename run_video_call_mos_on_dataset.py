import os
import pandas as pd
from vcm.model import VideoCallMosModel

csv_file = '/root/data/video_call_mos_set/video_call_mos_set.csv'
data_dir = '/root/data/video_call_mos_set/data'
results_dir = '/root/data/video_call_mos_set/results'
tmp_dir = '/root/data/video_call_mos_set/tmp'
checkpoint = 'vcm/video_call_mos_weights.pt'

vcm_model = VideoCallMosModel(checkpoint)

df = pd.read_csv(csv_file)
for index, row in df.iterrows():
    deg_video = os.path.join(data_dir, row['deg_video'])
    ref_video = os.path.join(data_dir, row['ref_video'])
    mos, results_csv = vcm_model(deg_video, ref_video, results_dir, tmp_dir, verbosity=0)
    print(f"{index} - MOS: {mos:0.2f}, deg_video {deg_video}, ref_video {ref_video}, results csv: {results_csv}")