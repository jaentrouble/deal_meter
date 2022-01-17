from assets import presets
from model_trainer import deal_model
import cv2
import json
import tensorflow as tf
import tensorflow.keras as keras
from skimage.metrics import structural_similarity as ssim
import numpy as np
import tqdm

DIGITS = 11
INPUT_SIZE = (640,64)

def hp_logger(
    vid_path,
    preset:presets.Preset,
    model_path,
):
    hp_model = keras.models.load_model(model_path)
    cap = cv2.VideoCapture(vid_path)
    icon = cv2.imread(preset.icon_name)
    ipos_1 = preset.icon_pos_hw[0]
    ipos_2 = preset.icon_pos_hw[1]
    hp_h_st = preset.hp_h_st
    hp_h_ed = preset.hp_h_ed
    hp_w_st = preset.hp_w_st
    hp_w_ed = preset.hp_w_ed
    hp_log = []
    digit_mul = 10**np.arange(DIGITS-1,-1,-1)

    f = 0 # frame count

    for _ in tqdm.trange(500):
        f += 1
        ret, frame = cap.read()
        if not ret:
            break
        fr_icon = frame[ipos_1[0]:ipos_2[0],ipos_1[1]:ipos_2[1]]
        if ssim(icon, fr_icon, channel_axis=2)<0.85:
            continue
        fr_hpbar = frame[hp_h_st:hp_h_ed,hp_w_st:hp_w_ed,:]
        fr_hpbar = cv2.resize(fr_hpbar, INPUT_SIZE)
        fr_hpbar = fr_hpbar[np.newaxis,:,:,:].astype(np.float32)
        out_vector = np.argmax(hp_model(fr_hpbar, training=False)[0],axis=-1)
        hp_pred = int(np.sum(digit_mul*out_vector))
        hp_log.append((f,hp_pred))
        
    
    with open(vid_path+'.log', 'w') as l:
        json.dump(hp_log, l)

if __name__ =='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--video',dest='video')
    args = parser.parse_args()

    hp_logger(
        args.video,
        presets.UHD110COM,
        'savedmodels/final'
    )
    