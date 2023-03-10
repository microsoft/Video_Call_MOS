#%%
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

from vcm.model import VcmNet, get_features
from vcm.training import VideoDataset, eval_model

def evaluate_and_plot(csv_file, video_idxs, data_dir, checkpoint, features, hidden_size, num_layers, bs):

    # model
    model = VcmNet(hidden_size=hidden_size, num_layers=num_layers, features=features)
    model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

    # dataset
    ds_val = VideoDataset(csv_file, data_dir, features, db_type='val')
    dl_val = DataLoader(ds_val, batch_size=bs, shuffle=False, drop_last=False, num_workers=os.cpu_count())

    # run model inference
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pcc_vcm, rmse_vcm, mos_pred_vcm = eval_model(model, dl_val, dev)

    # get (scaled) vmaf results
    mos_sub = ds_val.df['MOS'].to_numpy().reshape(-1)
    vmaf = ds_val.df['vmaf'].to_numpy().reshape(-1,1)
    lin_model = LinearRegression().fit(vmaf, mos_sub)
    mos_pred_vmaf = lin_model.predict(ds_val.df['vmaf'].to_numpy().reshape(-1,1))
    pcc_vmaf = pearsonr(mos_pred_vmaf, mos_sub)[0]
    rmse_vmaf = np.sqrt(np.mean((mos_pred_vmaf-mos_sub)**2))

    # print results
    print(f"VCM PCC: {pcc_vcm:0.2f}, VCM RMSE: {rmse_vcm:0.2f}, VMAF PCC: {pcc_vmaf:0.2f}, VMAF RMSE: {rmse_vmaf:0.2f}")

    # plot correlation diagrams
    plt.figure()
    plt.subplot(1,2,2)
    plt.plot(mos_sub, mos_pred_vcm, '.')
    plt.xlabel('MOS_P910')
    plt.ylabel('MOS_VCM')
    plt.ylim(0.9,5.1)
    plt.title(f"PCC {pcc_vcm:0.3f}, RMSE {rmse_vcm:0.3f}")
    plt.gca().set_aspect('equal')
    plt.subplot(1,2,1)
    plt.plot(mos_sub, mos_pred_vmaf, '.')
    plt.xlabel('MOS_P910')
    plt.ylabel('MOS_VMAF')
    plt.title(f"PCC {pcc_vmaf:0.3f}, RMSE {rmse_vmaf:0.3f}")
    plt.ylim(0.9,5.1)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

    # plot example predictions per frame
    for idx in video_idxs:
        df_feat = pd.read_csv(os.path.join(data_dir, ds_val.df['vmaf_results'].iloc[idx]))
        df_feat['vmaf_scaled'] = lin_model.predict(df_feat['vmaf'].to_numpy().reshape(-1,1))
        df_feat = get_features(df_feat, features)[1]   

        model.to(dev)
        model.eval()
        x, mos_sub = ds_val[idx]
        with torch.no_grad():
            mos_pred_vcm, frame_pred_vcm = model(x.unsqueeze(0).to(dev))
        mos_pred_vcm = mos_pred_vcm.item()
        mos_sub = mos_sub.item()
        frame_pred_vcm = frame_pred_vcm.to('cpu').numpy().reshape(-1)

        fps = 30
        t = np.arange(len(frame_pred_vcm)/fps, step=1/fps)
        plt.figure(figsize=(5.5, 2.5), dpi=300)    
        plt.plot([0, t[-1]], [mos_sub, mos_sub], 'k', label=f"MOS={mos_sub:0.2f}")
        plt.plot(t, frame_pred_vcm, label=f"VCM={mos_pred_vcm:0.2f}")
        plt.plot(t, df_feat['vmaf_scaled'], label=f"VMAF={df_feat['vmaf_scaled'].mean():0.2f}")
        plt.plot(t, df_feat['frame_skips']/15, label="Skip")
        plt.plot(t, df_feat['frame_freeze']/15,'-.', label="Freeze")
        plt.ylim([-0.1, 5.1])
        plt.xlabel('Time [s]')
        plt.ylabel('MOS')
        plt.title(f"Idx: {idx}, video: {ds_val.df['deg_video'].iloc[idx]}")
        plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left", ncol=1)    
        plt.show()