import ffmpeg
from PIL import Image, ImageFont, ImageDraw
import random
import cv2
from pathlib import Path
import tqdm
import json
import numpy as np

class VideoDSGenerator():
    MIN_BUF_SIZE = 20000

    def __init__(self, vid_dir:str) -> None:
        self.vid_path_list = list(Path(vid_dir).glob('*.mp4'))
        self.vid_log_list = []
        for vp in self.vid_path_list:
            with open(vp.with_suffix('.mp4.log'),'r') as j:
                self.vid_log_list.append(json.load(j))
        self.buffer_x = []
        self.buffer_y = []

    def refill_buffer(self):
        while len(self.buffer_x)< VideoDSGenerator.MIN_BUF_SIZE:
            tgt_idx = random.randrange(0,len(self.vid_path_list))
            cap = cv2.VideoCapture(str(self.vid_path_list[tgt_idx]))
            hp_log = self.vid_log_list[tgt_idx]
            for hp in hp_log:
                ret, frame = cap.read()
                if ret:
                    self.buffer_x.append(frame[:,:,::-1])
                    self.buffer_y.append(hp)
                else:
                    break
            cap.release()

    def __iter__(self):
        return self
    
    def __call__(self, *args):
        return self

    def __next__(self):
        if len(self.buffer_x)<VideoDSGenerator.MIN_BUF_SIZE:
            self.refill_buffer()
        idx = random.randrange(0,len(self.buffer_x))
        x = self.buffer_x.pop(idx)
        y = self.buffer_y.pop(idx)

        return x, y


def vid_dataset(vid_dir:str, max_digit:int, image_size:tuple[int,int], 
                batch_size:int):
    import tensorflow as tf
    dataset = tf.data.Dataset.from_generator(
        VideoDSGenerator(vid_dir),
        output_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.uint8),
            tf.TensorSpec(shape=[],dtype=tf.int64)
        )
    )
    def image_aug(image, raw_label):
        width= tf.cast(tf.shape(image)[1],tf.float32)
        w_st = tf.cast(width*0.27,tf.int32)
        w_ed = tf.cast(width*0.70,tf.int32)
        image = image[:,w_st:w_ed,:]
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.image.resize(image, image_size)
        # random invert color
        if tf.random.uniform([]) < 0.5:
            image = 1.0 - image
        # random shuffle rgb
        if tf.random.uniform([]) < 0.5:
            image = tf.gather(
                image,
                tf.random.shuffle([0,1,2]),
                axis=-1,
            )

        image = image * 255

        i = raw_label
        tf.debugging.assert_less(i,10**max_digit,
            message='Label is larger than max digit')
        d = tf.range(max_digit,0,-1,dtype=tf.int64)
        label = (i-(i//(10**d))*10**d)//(10**(d-1))

        return image, label

    dataset = dataset.map(image_aug)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()

    return dataset

def pool_func(_id):

    base_img_dir = 'videos/base'
    # init_digit = [11, 10, 9, 8, 7]
    # total_frames = [10000,1000,1000,1000,1000]
    # vid_num =    [100,10,10,10,10]
    init_digit = [11]
    total_frames = [100]
    vid_num =    [10]

    width = 540
    height = 30
    output_kwargs = {
        'vcodec' : 'libx264',
        # 'rc' : 'vbr_hq',
        # 'cq' : '18',
        # 'video_bitrate' : '30M',
        # 'profile:v' : 'high',
        # 'preset' : 'slow',
        'crf' : '30',
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
    if _id == 0:
        _iter = tqdm.tqdm(zip(init_digit, total_frames, vid_num))
    else:
        _iter = zip(init_digit, total_frames, vid_num)

    for d, a, n in _iter:
        if _id == 0:
            _range_iter = tqdm.trange(n)
            print('digit:',d)
        else:
            _range_iter = range(n)
        for i in _range_iter:
            vid_name = f'videos/vid_noise2/{_id}_{d}_{a}_{i}.mp4'
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
            if _id==0:
                tq = tqdm.trange(a,leave=False)
            else:
                tq = range(a)
            xy = (width//2+random.randrange(-10,11),
                  height//2+random.randrange(0,4))
            font = random.choice(fonts)

            hp_log = []
            for f in tq:
                new_img = random.choice(base_img_list).copy()
                new_img = new_img.resize((width,height))
                current_hp -= random.randrange(hp_step_st, hp_step_ed)
                if current_hp <= 0:
                    if _id==0:
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
    print('process done: ',_id)
            
if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(10) as p:
        p.map(pool_func,range(10))