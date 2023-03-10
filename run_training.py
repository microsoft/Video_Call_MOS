from vcm.training import train

# input / output data
csv_file = '/root/data/video_call_mos_set/video_call_mos_set.csv'
data_dir = '/root/data/video_call_mos_set/data'
output_dir = '/root/data/video_call_mos_set/output'

# training settings
epochs = 1000
features = ['frame_skips', 'frame_freeze', 'integer_motion', 'integer_adm2', 'integer_adm_scale0', 'integer_adm_scale1', 'integer_adm_scale2', 
            'integer_adm_scale3', 'integer_vif_scale0', 'integer_vif_scale1', 'integer_vif_scale2', 'integer_vif_scale3']
hidden_size = 256
num_layers = 6
bs = 128
lr = 1e-4
lr_patience = 20

if __name__ == "__main__":
    train(csv_file, data_dir, output_dir, epochs, features, hidden_size, num_layers, bs, lr, lr_patience)