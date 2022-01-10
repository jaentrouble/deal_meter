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
    output_kwargs = {
        'vcodec' : 'h264_nvenc',
        'rc:v' : 'vbr',
        'cq:v' : '18',
        'video_bitrate' : '3K',
        'preset' : 'slow',
        'profile:v' : 'medium',
        'r' : '60',
        's' : f'{width}x{height}',
    }

    base_img_list = [
        Image.open(d) for d in Path(base_img_dir).iterdir()
            if d.match('*.png')
    ]
    fonts = [
        ImageFont.truetype('NanumBarunGothic.ttf',size=s)
            for s in range(28,36)
    ]

    for d, a, n in tqdm.tqdm(zip(init_digit, total_frames, vid_num)):
        for i in tqdm.trange(n):
            vid_name = f'videos/vid_noise/{d}_{a}_{i}.mp4'
            log_name = vid_name + '.log'
            process = (
                ffmpeg
                    .input('pipe:', format='rawvideo', pix_fmt='rgb24', 
                            s=f'{width}x{height}')
                    .output(vid_name,**output_kwargs)
                    .overwrite_output()
                    .run_async(pipe_stdin=True)
            )
            start_hp = random.randrange(10**(d-1),10**d)
            current_hp = start_hp
            hp_step_st = (10**d // a)//2
            hp_step_ed = hp_step_st * 3
            tq = tqdm.trange(a,leave=False)
            xy = (new_img.width//2+random.randrange(-50,51),
                  new_img.height//2+random.randrange(0,16))

            hp_log = []
            for f in tq:
                new_img = random.choice(base_img_list).copy()
                current_hp -= random.randrange(hp_step_st, hp_step_ed)
                hp_log.append(current_hp)
                if current_hp <= 0:
                    tq.close()
                    break
                hp_text = f'{current_hp}/{start_hp}'
                draw = ImageDraw.Draw(new_img)
                font = random.choice(fonts)
                draw.text(
                    xy=xy,
                    text=hp_text,
                    fill=(255,255,255),
                    font=font,
                    anchor='mm'
                )
                new_img.resize((width,height))
                new_frame = np.array(new_img.convert('RGB'),dtype=np.uint8)
                process.stdin.write(
                    new_frame.tobytes()
                )
            process.stdin.close()
            process.wait()
            with open(log_name,'w') as log_file:
                json.dump(hp_log,log_file)
            
