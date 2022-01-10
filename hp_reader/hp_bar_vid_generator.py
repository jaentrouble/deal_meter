import ffmpeg
from PIL import Image, ImageFont, ImageDraw
import random
import cv2
from pathlib import Path
import tqdm
import json
import numpy as np

if __name__ == '__main__':

    base_img_dir = 'videos/base'
    # init_digit = [11, 10, 9, 8, 7]
    # total_frames = [100000,10000,10000,10000,10000]
    # vid_num =    [1000,100,100,100,100]
    init_digit = [11]
    total_frames = [100]
    vid_num =    [10]

    width = 540
    height = 30
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    base_img_list = [
        Image.open(d) for d in Path(base_img_dir).iterdir()
            if d.match('*.png')
    ]
    fonts = [
        ImageFont.truetype('NanumBarunGothic.ttf',size=s)
            for s in range(13,16)
    ]

    for d, a, n in tqdm.tqdm(zip(init_digit, total_frames, vid_num)):
        for i in tqdm.trange(n):
            vid_name = f'videos/vid_noise/{d}_{a}_{i}.mp4'
            log_name = vid_name + '.log'
            video_writer = cv2.VideoWriter(
                vid_name,
                fourcc,
                60,
                (width,height),
            )
            start_hp = random.randrange(10**(d-1),10**d)
            current_hp = start_hp
            hp_step_st = (10**d // a)//2
            hp_step_ed = hp_step_st * 3
            tq = tqdm.trange(a,leave=False)
            xy = (width//2+random.randrange(-10,11),
                  height//2+random.randrange(0,4))

            hp_log = []
            for f in tq:
                new_img = base_img_list[0].copy()
                new_img.resize((width,height))
                current_hp -= random.randrange(hp_step_st, hp_step_ed)
                if current_hp <= 0:
                    tq.close()
                    break
                hp_log.append(current_hp)
                hp_text = f'{current_hp}/{start_hp}'
                draw = ImageDraw.Draw(new_img)
                font = fonts[0]
                draw.text(
                    xy=xy,
                    text=hp_text,
                    fill=(255,255,255),
                    font=font,
                    anchor='mm'
                )
                
                new_frame = np.array(new_img.convert('RGB'),dtype=np.uint8)
                new_frame = cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(new_frame)
            video_writer.release()
            with open(log_name,'w') as log_file:
                json.dump(hp_log,log_file)
            
