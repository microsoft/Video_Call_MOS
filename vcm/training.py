import torch
import os
import time
import datetime
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from vcm.model import VcmNet, get_features

# evaluation function
def eval_model(model, dl_val, dev):
    model.to(dev)
    model.eval()
    with torch.no_grad():
        y_hat, y = zip(*[(model(xb.to(dev))[0].cpu().numpy(), yb.view(-1)) for xb, yb in dl_val])
    model.train()
    y_hat, y = np.concatenate(y_hat), np.concatenate(y)
    pcc_val, rmse = pearsonr(y_hat, y)[0], np.sqrt(np.mean((y_hat - y) ** 2))
    return pcc_val, rmse, y_hat

# dataset
class VideoDataset(Dataset):
    def __init__(self, csv_file, data_dir, features, db_type):
        self.features = features
        df = pd.read_csv(csv_file)
        self.df = df[df['type']==db_type]
        self.data_dir = data_dir
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        df_features = pd.read_csv(os.path.join(self.data_dir, self.df['vmaf_results'].iloc[idx]))
        x = get_features(df_features, self.features)[0]
        y = self.df['MOS'].iloc[idx]
        x = torch.as_tensor(x, dtype=torch.float)
        y = torch.as_tensor(y, dtype=torch.float).view(-1)        
        return x, y
    
# train function
def train(csv_file, data_dir, output_dir, epochs, features, hidden_size, num_layers, bs, lr, lr_patience):

    # model
    model = VcmNet(hidden_size=hidden_size, num_layers=num_layers, features=features)

    ds_train = VideoDataset(csv_file, data_dir, features, db_type='train')
    ds_val = VideoDataset(csv_file, data_dir, features, db_type='val')
    dl_train = DataLoader(ds_train, batch_size=bs, shuffle=True, drop_last=True, num_workers=os.cpu_count())
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, drop_last=False, num_workers=os.cpu_count())

    # optimizer
    opt = optim.Adam(model.parameters(), lr=lr)      
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', verbose=True, threshold=0.0001, patience=lr_patience)

    # initialize train loop
    best_pcc = 0
    best_rmse = 1
    best_rmse_path = ''
    best_pcc_path = ''
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(dev)
    now = datetime.datetime.today()
    runname = now.strftime("%y%m%d_%H%M%S%f")
    run_dir = os.path.join(output_dir, runname)
    os.makedirs(run_dir, exist_ok=True)

    # training loop
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        tic = time.time()
        for batch_cnt, (x, y) in enumerate(dl_train, 1):
            x = x.to(dev)
            y = y.to(dev).view(-1)
            opt.zero_grad()
            y_hat = model(x)[0]
            loss = criterion(y_hat, y)
            loss.backward()
            opt.step()
            torch.save(model.state_dict(), os.path.join(run_dir, 'last_checkpoint.pt'))
            running_loss += loss.item()
        running_loss = running_loss/batch_cnt
        scheduler.step(running_loss)

        # evaluate model
        pcc_train, rmse_train, _ = eval_model(model, dl_train, dev)
        pcc_val, rmse_val, _ = eval_model(model, dl_val, dev)
        print(f"epoch {epoch:04d}, runtime {time.time()-tic:0.2f}s, loss {running_loss:0.4f}, pcc_train {pcc_train:0.4f}, rmse_train {rmse_train:0.4f}, pcc_val {pcc_val:0.4f}, rmse_val {rmse_val:0.4f}")

        # save checkpoint
        if pcc_val>best_pcc:
            if os.path.exists(best_pcc_path):
                os.remove(best_pcc_path)    
            best_pcc_path = os.path.join(run_dir, f"best_pcc_ep_{epoch}_pcc_{pcc_val:0.3f}_rmse_{rmse_val:0.3f}".replace('.', '')+".pt")
            torch.save(model.state_dict(), best_pcc_path)
            print(f'  saved checkpoint: {best_pcc_path}')
            best_pcc = pcc_val
        if rmse_val<best_rmse:
            if os.path.exists(best_rmse_path):
                os.remove(best_rmse_path)
            best_rmse_path = os.path.join(run_dir, f"best_rmse_ep_{epoch}_pcc_{pcc_val:0.3f}_rmse_{rmse_val:0.3f}".replace('.', '')+".pt")
            torch.save(model.state_dict(), best_rmse_path)
            print(f'  saved checkpoint: {best_rmse_path}')
            best_rmse = rmse_val

    print('Finished Training')