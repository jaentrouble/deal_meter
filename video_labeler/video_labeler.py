import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np

CUT_IDLE = 0
CUT_WAITING1 = 1
CUT_WAITING2 = 2

class Console():
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title('Video Labeler')
        self.mainframe = ttk.Frame(self.root)
        self.mainframe.grid(column=0,row=0)

        self.label_image = ttk.Label(self.mainframe)
        self.label_image.grid(column=0,row=0,columnspan=2,rowspan=3)
        self.label_image.bind('<Button-1>', self.image_click_event_handler)


        # TODO : Fill commands
        self.button_cut = ttk.Button(
            self.mainframe,
            text='Cut',
            command=self.button_cut_command
        )
        self.button_cut.grid(column=0, row=3)
        self.cut_mode = CUT_IDLE
        
        self.button_reset = ttk.Button(
            self.mainframe,
            text='Reset',
            # command=
        )
        self.button_reset.grid(column=1, row=3)

        self.radio_var = tk.IntVar(value=0)

        self.radio_nonlabel = ttk.Radiobutton(
            self.mainframe,
            text='',
            variable=self.radio_var,
            value=0,
        )
        self.radio_nonlabel.grid(column=2,row=0)
        self.label_nonlabel = ttk.Label(
            self.mainframe,
            text='Not-labelable'
        )
        self.label_nonlabel.grid(column=3,row=0)
        
        self.radio_label = ttk.Radiobutton(
            self.mainframe,
            text='',
            variable=self.radio_var,
            value=1,
        )
        self.radio_label.grid(column=2,row=1)

        self.entry_var = tk.StringVar()
        self.entry_label = ttk.Entry(
            self.mainframe,
            textvariable=self.entry_var,
        )
        self.entry_label.grid(column=3,row=1)

        self.label_info = ttk.Label(
            self.mainframe,
            text='Save: Enter\nNext: F\nPrev: D'
        )
        self.label_info.grid(column=3, row=2)

        self.button_open = ttk.Button(
            self.mainframe,
            text='Open',
            command=self.load_vid
        )
        self.button_open.grid(column=4,row=3)
    
    def load_vid(self):
        self.vid_name = filedialog.askopenfilename()
        print(self.vid_name)
        if self.vid_name=='':
            return
        cap = cv2.VideoCapture(self.vid_name)
        self.frames = []
        while (cap.isOpened()):
            ret,frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame)
            else:
                break
        cap.release()

        self.cut_point_1 = None
        self.cut_point_2 = None
        self.frame_idx = 0

    def button_cut_command(self):
        self.cut_mode = CUT_WAITING1
        self.cut_point_1 = None
        self.cut_point_2 = None
        self.update()

    def image_click_event_handler(self,event:tk.Event):
        # TODO : Just for checking
        # self.entry_var.set(str(event.x)+'/'+str(event.y))
        if self.cut_mode == CUT_WAITING1:
            self.cut_point_1 = (event.y,event.x)
            self.cut_mode = CUT_WAITING2
            self.update()
        elif self.cut_mode == CUT_WAITING2:
            new_point1 = (min(self.cut_point_1[0],event.y),
                          min(self.cut_point_1[1],event.x))
            new_point2 = (max(self.cut_point_1[0],event.y),
                          max(self.cut_point_1[1],event.x))
            self.cut_point_1 = new_point1
            self.cut_point_2 = new_point2
            self.cut_mode = CUT_IDLE
            self.update()

    def update(self):
        cur_raw_frame = self.frames[self.frame_idx].copy()
        if self.cut_point_2 is None:
            self.current_image = ImageTk.PhotoImage(Image.fromarray(
                cur_raw_frame
            ))
        else:
            self.current_image = ImageTk.PhotoImage(Image.fromarray(
                cur_raw_frame[self.cut_point_1[0]:self.cut_point_2[0],
                              self.cut_point_1[1]:self.cut_point_2[1]]
            ))
        
        self.label_image.configure(image=self.current_image)

    def run(self):
        self.load_vid()
        self.update()
        self.root.mainloop()
        


if __name__ == '__main__':
    Console().run()