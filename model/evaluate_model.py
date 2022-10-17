'''
Script for analysing video mos model 
'''

import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
import time
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
from sklearn.linear_model import LinearRegression
import onnxruntime

all_features = [
    'Frame', 'integer_motion2', 'integer_motion', 'integer_adm2',
    'integer_adm_scale0', 'integer_adm_scale1', 'integer_adm_scale2',
    'integer_adm_scale3', 'psnr_y', 'psnr_cb', 'psnr_cr', 'float_ssim',
    'integer_vif_scale0', 'integer_vif_scale1', 'integer_vif_scale2',
    'integer_vif_scale3', 'vmaf', 'ref_frames'
    ]

seq_len = 181
csv_file = "C:/data/set_1/set_1.csv"
data_dir = "C:/data/set_1/data"
model_path = "C:/data/model.pt"

# train settings (must be same as during training) ----------
features = [
    'ref_frame_diff', 'freeze_duration', 'integer_motion',
    'integer_adm2',
    'integer_adm_scale0', 'integer_adm_scale1', 'integer_adm_scale2',
    'integer_adm_scale3', 
    'integer_vif_scale0', 'integer_vif_scale1', 'integer_vif_scale2',
    'integer_vif_scale3', 
    ]

hidden_size = 256
num_layers = 6
num_workers = 0
bs = 128

# start inference --------------------------------------------
print(f"features {features}")

if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("use cuda")
else:
    dev = torch.device("cpu")
    print("use cpu")

def eval_model(model, dl_val):
    model.eval()
    model.to(dev)
    with torch.no_grad():
        y_hat_list = [ [model(xb.to(dev))[0].cpu().numpy(), yb.cpu().numpy()] for xb, yb, idx in dl_val]
    model.train()
    yy = np.concatenate(y_hat_list, axis=1)
    y_hat = yy[0,:,0].reshape(-1)
    y = yy[1,:,0].reshape(-1)
    pcc_val, _ = pearsonr(y_hat, y)
    rmse = np.sqrt(np.mean((y_hat-y)**2))
    return pcc_val, rmse, y, y_hat

class VideoDataset(Dataset):
    def __init__(self, csv_file, data_dir, features, db_type):
        self.features = features
        df = pd.read_csv(csv_file)
        self.df = df[df['db_type']==db_type]
        self.data_dir = data_dir
    def __len__(self):
        return len(self.df)
    def get_freeze_time(self, x):
        xold=0
        xfreeze = 0
        xfreezes = []
        for x1 in x:
            if x1==xold:
                xfreeze+=1
            else:
                xfreeze=0
            xold=x1
            xfreezes.append(xfreeze)
        xfreezes = np.array(xfreezes)
        return xfreezes        
    def __getitem__(self, idx):
        df_features = pd.read_csv(os.path.join(self.data_dir, self.df.vmaf_csv.iloc[idx]))
        df_features['ref_frame_diff'] = df_features['ref_frames'].diff().fillna(0)
        df_features['ref_frames_2'] = (df_features['ref_frames'] - df_features['ref_frames'].iloc[0]) / len(df_features['ref_frames'])
        df_features['ref_frames_3'] = df_features['ref_frames'] - df_features['ref_frames'].iloc[0]
        df_features['ref_frame_diff_2'] = df_features['ref_frames_2'].diff().fillna(0)
        df_features['freeze_duration'] = self.get_freeze_time(df_features['ref_frames'].to_numpy())        
        x = df_features[features].to_numpy()
        y = self.df.MOS.iloc[idx]
        x = torch.as_tensor(x, dtype=torch.float)
        y = torch.as_tensor(y, dtype=torch.float).view(-1)
        return x, y, idx

ds_train = VideoDataset(csv_file, data_dir, features, db_type='train')
ds_val = VideoDataset(csv_file, data_dir, features, db_type='val')

pcc_train, _ = pearsonr(ds_train.df.vmaf_clip.to_numpy().reshape(-1), ds_train.df.MOS.to_numpy().reshape(-1))
pcc_val, _ = pearsonr(ds_val.df.vmaf_clip.to_numpy().reshape(-1), ds_val.df.MOS.to_numpy().reshape(-1))

print(f"len train {len(ds_train)}, len val {len(ds_val)}")
print(f"vmaf pcc_train {pcc_train:0.3f}, vmaf pcc_val {pcc_val:0.3f}")
print()

dl_train = DataLoader(ds_train,
                batch_size=bs,
                shuffle=True,
                drop_last=True,
                pin_memory=False,
                num_workers=num_workers)

dl_val = DataLoader(ds_val,
                batch_size=bs,
                shuffle=False,
                drop_last=False,
                pin_memory=False,
                num_workers=num_workers)
        
class Net(nn.Module):
    def __init__(self, in_feat, hidden_size=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.BatchNorm1d(in_feat)
        self.lstm = nn.LSTM(in_feat, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.pad = 100

    def forward(self, x, n_seq=None):    
        if n_seq:
            x = x[:,:n_seq,:]
        x = self.layer_norm(x.transpose(2,1)).transpose(2,1)
        x = F.pad(x, (0,0,self.pad,0), "reflect")
        x, _ = self.lstm(x)
        x = x[:,self.pad:,:]
        x = self.fc(x)
        x_step = x
        x = x.mean(1)
        return x, x_step
        
in_feat=len(features)
model = Net(in_feat, hidden_size, num_layers)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

#%% Evaluate model
pcc_val, rmse_val, y, y_hat = eval_model(model, dl_val)


plt.figure(figsize=(3.0, 3.0), dpi=300)
plt.plot(y_hat, y, '.')
plt.xlabel('Predicted MOS')
plt.ylabel('Subjective MOS')
plt.ylim(0.9,5.1)
plt.xlim(0.9,5.1)
plt.xticks([1,2,3,4,5])
plt.title(f"PCC: {pcc_val:0.3f}, RMSE: {rmse_val:0.3f}")
plt.gca().set_aspect('equal')
plt.savefig("results_lstm.pdf", format="pdf", dpi=300, bbox_inches="tight")


plt.figure()

plt.subplot(1,2,2)
plt.plot(y, y_hat, '.')
plt.xlabel('MOS_Mturk')
plt.ylabel('MOS_LSTM')
plt.ylim(0.9,5.1)
plt.title(f"pcc {pcc_val:0.3f}, rmse {rmse_val:0.3f}")
plt.gca().set_aspect('equal')

x = ds_train.df.vmaf_clip.to_numpy().reshape(-1,1)
y = ds_train.df.MOS.to_numpy().reshape(-1)
lin_model = LinearRegression().fit(x, y)

vmaf_scaled = lin_model.predict(ds_val.df['vmaf_clip'].to_numpy().reshape(-1,1))
y = ds_val.df['MOS']
pcc_val, _ = pearsonr(vmaf_scaled, y)
rmse_val = np.sqrt(np.mean((vmaf_scaled-y)**2))

plt.subplot(1,2,1)
plt.plot(y, vmaf_scaled, '.')
plt.xlabel('MOS_Mturk')
plt.ylabel('MOS_VMAF')
plt.title(f"pcc {pcc_val:0.3f}, rmse {rmse_val:0.3f}")
plt.ylim(0.9,5.1)
plt.gca().set_aspect('equal')
plt.tight_layout()



#%% Plot MOS Per Frame
def get_freeze_time(x):
    xold=0
    xfreeze = 0
    xfreezes = []
    for x1 in x:
        if x1==xold:
            xfreeze+=1
        else:
            xfreeze=0
        xold=x1
        xfreezes.append(xfreeze)
    xfreezes = np.array(xfreezes)
    xfreezes = np.maximum(0,xfreezes-1)
    return xfreezes

idxs = [127, 128,  275]
# idxs = range(120, 150)

for idx in idxs:
    df_vmaf = pd.read_csv(os.path.join(data_dir, ds_val.df.vmaf_csv.iloc[idx]))
    df_vmaf['vmaf_scaled'] = lin_model.predict(df_vmaf['vmaf'].to_numpy().reshape(-1,1))
    
    ref_diff = (np.maximum(df_vmaf['ref_frames'].diff()-2,0))/10
    x = df_vmaf['ref_frames'].to_numpy()
    
    freeze_time = get_freeze_time(x)

    fps = 30
    model.to(dev)
    model.eval()
    x, y, idx = ds_val[idx]
    with torch.no_grad():
        y_hat, y_hat_step = model(x.unsqueeze(0).to(dev))
    y_hat = y_hat.item()
    y = y.item()
    y_hat_step = y_hat_step.to('cpu').numpy().reshape(-1)
    t = np.arange(len(y_hat_step)/fps, step=1/fps)
    plt.figure()
    plt.plot([0, t[-1]], [y,y], 'k', label='MOS')
    plt.plot(t, y_hat_step, label="LSTM")
    plt.plot(t, df_vmaf['vmaf_scaled'], label="VMAF")
    plt.plot(t, ref_diff, label="Skip")
    plt.plot(t, freeze_time/10,'-.', label="Freeze")
    
    plt.ylim([-0.1, 5.1])
    plt.xlabel('time [s]')
    plt.title(f"Idx: {idx}, Profile: {ds_val.df.NetEmProfile.iloc[idx]}")
    plt.legend(bbox_to_anchor=(1.00, 1), loc="upper left", ncol=1)
    plt.show()
    
    
        
    plt.figure(figsize=(5.5, 2.5), dpi=300)    
    plt.plot([0, t[-1]], [y,y], 'k', label='MOS')
    plt.plot(t, y_hat_step, label="LSTM")
    plt.plot(t, df_vmaf['vmaf_scaled'], label="VMAF")
    plt.plot(t, ref_diff, label="Skip")
    plt.plot(t, freeze_time/10,'-.', label="Freeze")
    plt.ylim([-0.1, 5.1])
    plt.xlabel('Time [s]')
    plt.ylabel('MOS')
    plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left", ncol=1)    
    plt.savefig(f"example_{idx}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
