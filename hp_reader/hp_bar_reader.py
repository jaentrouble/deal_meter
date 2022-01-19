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
MAX_BUF = 36000

def hp_logger(
    vid_path,
    frames,
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
    f_log = []
    digit_mul = 10**np.arange(DIGITS-1,-1,-1)

    f = 0 # frame count
    
    input_buf = np.empty((frames,INPUT_SIZE[1],INPUT_SIZE[0],3),
                         dtype=np.float32)
    buf_idx = 0
    for _ in tqdm.trange(frames):
        f += 1
        ret, frame = cap.read()
        if not ret:
            break
        fr_icon = frame[ipos_1[0]:ipos_2[0],ipos_1[1]:ipos_2[1]]
        if ssim(icon, fr_icon, channel_axis=2)<0.85:
            continue
        fr_hpbar = frame[hp_h_st:hp_h_ed,hp_w_st:hp_w_ed,:]
        fr_hpbar = cv2.resize(fr_hpbar, INPUT_SIZE)
        if buf_idx == 0:
            cv2.imwrite(vid_path+'_sample.png',fr_hpbar)
        input_buf[buf_idx] = fr_hpbar.astype(np.float32)
        buf_idx += 1
        f_log.append(f)

    done_idx = 0
    while done_idx < buf_idx:
        input_tensor = input_buf[done_idx:min(buf_idx,MAX_BUF+done_idx)]
        out_logit = hp_model.predict(
                    input_tensor,
                    verbose=1
        )
        out_vector = np.argmax(out_logit,axis=-1)
        hp_pred = np.sum(digit_mul*out_vector,axis=-1)
        for p in hp_pred:
            hp_log.append(int(p))
        done_idx += MAX_BUF
        
    
    assert len(hp_log) == len(f_log)
        
    with open(vid_path+'.log', 'w') as l:
        json.dump([f_log,hp_log], l)

if __name__ =='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--video',dest='video')
    parser.add_argument('-f','--frames',dest='frames',type=int)
    args = parser.parse_args()

    hp_logger(
        args.video,
        args.frames,
        presets.FHD100COM,
        'savedmodels/final'
    )
    