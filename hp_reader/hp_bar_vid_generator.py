import ffmpeg
from PIL import Image, ImageFont, ImageDraw
import random
import cv2
from pathlib import Path
import tqdm
import json
import numpy as np


def pool_func(_id):

    base_img_dir = 'videos/base'
    init_digit = [11, 10, 9, 8, 7]
    total_frames = [100000,10000,10000,10000,10000]
    vid_num =    [100,10,10,10,10]
    # init_digit = [11]
    # total_frames = [100]
    # vid_num =    [10]

    width = 540
    height = 30
    output_kwargs = {
        'vcodec' : 'libx264',
        # 'rc' : 'vbr_hq',
        # 'cq' : '18',
        # 'video_bitrate' : '30M',
        # 'profile:v' : 'high',
        # 'preset' : 'slow',
        'crf' : '26',
        'pix_fmt' : 'yuv420p',
        'r' : 60,
        's' : f'{width}x{height}'
    }

    base_img_list = [
        Image.open(d) for d in Path(base_img_dir).iterdir()
            if d.match('*.png')
    ]
    fonts = [
        ImageFont.truetype('NanumBarunGothic.ttf',size=s)
            for s in range(14,18)
    ]

    for d, a, n in tqdm.tqdm(zip(init_digit, total_frames, vid_num)):
        print('digit:',d)
        for i in tqdm.trange(n):
            vid_name = f'videos/vid_noise/{_id}_{d}_{a}_{i}.mp4'
            log_name = vid_name + '.log'
            process = (
                ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24', 
                            r=60,s=f'{width}x{height}')
                    .output(vid_name,**output_kwargs)
                    .overwrite_output()
                    .run_async(pipe_stdin=True,quiet=True)
            )
            start_hp = random.randrange(10**(d-1),10**d)
            current_hp = start_hp
            hp_step_st = (10**d // a)//2
            hp_step_ed = hp_step_st * 3
            tq = tqdm.trange(a,leave=False)
            xy = (width//2+random.randrange(-10,11),
                  height//2+random.randrange(0,4))
            font = random.choice(fonts)

            hp_log = []
            for f in tq:
                new_img = random.choice(base_img_list).copy()
                new_img = new_img.resize((width,height))
                current_hp -= random.randrange(hp_step_st, hp_step_ed)
                if current_hp <= 0:
                    tq.close()
                    break
                hp_log.append(current_hp)
                hp_text = f'{current_hp}/{start_hp}'
                draw = ImageDraw.Draw(new_img)
                
                draw.text(
                    xy=xy,
                    text=hp_text,
                    fill=(255,255,255),
                    font=font,
                    anchor='mm'
                )
                
                new_frame = np.array(new_img.convert('RGB'),dtype=np.uint8)
                process.stdin.write(
                    new_frame.tobytes()
                )
            process.stdin.close()
            process.wait()
            with open(log_name,'w') as log_file:
                json.dump(hp_log,log_file)
            
if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(10) as p:
        p.map(pool_func,range(10))