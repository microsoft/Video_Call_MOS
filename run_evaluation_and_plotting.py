from vcm.evaluation_and_plotting import evaluate_and_plot

# input / output data
data_dir = '/root/data/video_call_mos_set/data'
csv_file = '/root/data/video_call_mos_set/video_call_mos_set.csv'

# csv index for which videos to plot per frame results
video_idxs = [30, 65, 117, 164, 169, 197] 

# model settings (must be same as during training)
checkpoint = 'vcm/video_call_mos_weights.pt'
features = ['frame_skips', 'frame_freeze', 'integer_motion', 'integer_adm2', 'integer_adm_scale0', 'integer_adm_scale1', 'integer_adm_scale2', 
            'integer_adm_scale3', 'integer_vif_scale0', 'integer_vif_scale1', 'integer_vif_scale2', 'integer_vif_scale3']
hidden_size = 256
num_layers = 6
bs = 128

if __name__ == "__main__":
    evaluate_and_plot(csv_file, video_idxs, data_dir, checkpoint, features, hidden_size, num_layers, bs)