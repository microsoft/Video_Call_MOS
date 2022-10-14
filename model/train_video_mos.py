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
import datetime
import shutil


all_features = [
    'Frame', 'integer_motion2', 'integer_motion', 'integer_adm2',
    'integer_adm_scale0', 'integer_adm_scale1', 'integer_adm_scale2',
    'integer_adm_scale3', 'psnr_y', 'psnr_cb', 'psnr_cr', 'float_ssim',
    'integer_vif_scale0', 'integer_vif_scale1', 'integer_vif_scale2',
    'integer_vif_scale3', 'vmaf', 'ref_frames'
    ]
    

seq_len = 181
csv_file = "/mnt/data/set_1.csv"
data_dir = "/mnt/data/set_1/"
output_dir = '/mnt/output/'

# train settings
    
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
    
bs = 128
num_workers = 12
lr = 1e-4
lr_patience = 20


# start 
print(f"features {features}")


if torch.cuda.is_available():
    dev = torch.device("cuda")
    print("use cuda")
else:
    dev = torch.device("cpu")
    print("use cpu")

def eval_model(model, dl_val):
    model.eval()
    with torch.no_grad():
        y_hat_list = [ [model(xb.to(dev)).cpu().numpy(), yb.cpu().numpy()] for xb, yb, idx in dl_val]
    model.train()
    yy = np.concatenate(y_hat_list, axis=1)
    y_hat = yy[0,:,0].reshape(-1)
    y = yy[1,:,0].reshape(-1)
    pcc_val, _ = pearsonr(y_hat, y)
    rmse = np.sqrt(np.mean((y_hat-y)**2))
    return pcc_val, rmse


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
        # xfreezes = np.maximum(0,xfreezes-1)
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
        
    def forward(self, x):        
        x = self.layer_norm(x.transpose(2,1)).transpose(2,1)
        x = F.pad(x, (0,0,self.pad,0), "reflect")
        x, _ = self.lstm(x)
        x = x[:,self.pad:,:]
        x = self.fc(x)
        x_step = x
        x = x.mean(1)
        return x

in_feat=len(features)
model = Net(in_feat, hidden_size, num_layers)
opt = optim.Adam(model.parameters(), lr=lr)      
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        'min',
        verbose=True,
        threshold=0.0001,
        patience=lr_patience)


# training loop
epochs = 1000
best_pcc = 0
best_rmse = 1
best_rmse_path = ''
best_pcc_path = ''
model.to(dev)
now = datetime.datetime.today()
runname = now.strftime("%y%m%d_%H%M%S%f")
run_dir = os.path.join(output_dir, runname)
os.makedirs(run_dir, exist_ok = True)
shutil.copyfile('train_video_mos.py', os.path.join(run_dir, 'train_video_mos.py'))
for epoch in range(1, epochs+1):  # loop over the dataset multiple times

    running_loss = 0.0
    y_train_hat = np.zeros((len(ds_train), 1))
    tic = time.time()
    for batch_cnt, (x, y, idx) in enumerate(dl_train, 1):

        # move to gpu
        x = x.to(dev)
        y = y.to(dev)

        # zero the parameter gradients
        opt.zero_grad()

        # forward + backward + optimize
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        opt.step()
        
        torch.save(model.state_dict(), os.path.join(run_dir, 'last_checkpoint.pt'))

        y_train_hat[idx] = y_hat.detach().cpu().numpy()

        running_loss += loss.item()
    running_loss = running_loss/batch_cnt
    # pcc_train, _ = pearsonr(y_train_hat.reshape(-1), ds_train.df.MOS.to_numpy().reshape(-1))
    pcc_train, rmse_train = eval_model(model, dl_train)
    pcc_val, rmse_val = eval_model(model, dl_val)
    print(f"epoch {epoch:04d}, runtime {time.time()-tic:0.2f}s, loss {running_loss:0.4f}, pcc_train {pcc_train:0.4f}, rmse_train {rmse_train:0.4f}, pcc_val {pcc_val:0.4f}, rmse_val {rmse_val:0.4f}")

    if pcc_val>best_pcc:
        if os.path.exists(best_pcc_path):
            os.remove(best_pcc_path)    
        best_pcc_name = f'best_pcc_ep_{epoch}_pcc_{pcc_val:0.3f}_rmse_{rmse_val:0.3f}'.replace('.','')
        best_pcc_path = os.path.join(run_dir, f'{best_pcc_name}.pt')
        torch.save(model.state_dict(), best_pcc_path)
        print(f'saved {best_pcc_name}')
        best_pcc = pcc_val

    if rmse_val<best_rmse:
        if os.path.exists(best_rmse_path):
            os.remove(best_rmse_path)
        best_rmse_name = f'best_rmse_ep_{epoch}_pcc_{pcc_val:0.3f}_rmse_{rmse_val:0.3f}'.replace('.','')
        best_rmse_path = os.path.join(run_dir, f'{best_rmse_name}.pt')
        torch.save(model.state_dict(), best_rmse_path)
        print(f'saved {best_rmse_name}')
        best_rmse = rmse_val
        

    scheduler.step(running_loss)
print('Finished Training')
