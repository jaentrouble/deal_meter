import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import cv2
from pathlib import Path

CUT_IDLE = 0
CUT_WAITING1 = 1
CUT_WAITING2 = 2

RADIO_NON = 0
RADIO_LABEL = 1

NO_LABEL = 'NL'

class Console():
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title('Video Labeler')
        self.root.bind('<Key-f>',self.f_click_event_handler)
        self.root.bind('<Return>',self.f_click_event_handler)
        self.root.bind('<Key-d>',self.d_click_event_handler)
        self.mainframe = ttk.Frame(self.root)
        self.mainframe.grid(column=0,row=0)
        

        self.label_image = ttk.Label(self.mainframe)
        self.label_image.grid(column=0,row=0,columnspan=2,rowspan=3)
        self.label_image.bind('<Button-1>', self.image_click_event_handler)
        

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
            command=self.button_reset_command
        )
        self.button_reset.grid(column=1, row=3)

        self.radio_var = tk.IntVar(value=0)

        self.radio_nonlabel = ttk.Radiobutton(
            self.mainframe,
            text='',
            variable=self.radio_var,
            value=0,
            command=self.radio_button_command,
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
            command=self.radio_button_command,
        )
        self.radio_label.grid(column=2,row=1)

        self.entry_var = tk.StringVar()
        self.entry_label = ttk.Entry(
            self.mainframe,
            textvariable=self.entry_var,
            state='disabled',
        )
        self.entry_label.grid(column=3,row=1)

        self.label_info = ttk.Label(
            self.mainframe,
            text='Next: F or Enter\nPrev: D'
        )
        self.label_info.grid(column=3, row=2)

        self.button_useless = ttk.Button(
            self.mainframe,
            text='Nothing'
        )
        self.button_useless.grid(column=4, row=1)

        self.index_var = tk.StringVar(value=str(0))
        self.label_index = ttk.Label(
            self.mainframe,
            textvariable=self.index_var
        )
        self.label_index.grid(column=4,row=2)

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
        self.labels = []
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
        self.update()

    def radio_button_command(self):
        if self.radio_var.get() == RADIO_NON:
            self.entry_label.state(['disabled'])
        elif self.radio_var.get() == RADIO_LABEL:
            self.entry_label.state(['!disabled'])

    def button_cut_command(self):
        self.cut_mode = CUT_WAITING1
        self.cut_point_1 = None
        self.cut_point_2 = None
        self.update()

    def button_reset_command(self):
        self.cut_mode = CUT_IDLE
        self.cut_point_1 = None
        self.cut_point_2 = None
        self.update()

    def image_click_event_handler(self,event:tk.Event):
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

    def f_click_event_handler(self, event:tk.Event):
        """
        [Next] key
        """
        if self.radio_var.get()==RADIO_NON:
            new_label = NO_LABEL
        elif self.radio_var.get()==RADIO_LABEL:
            try:
                new_label = int(self.entry_var.get())
            except ValueError:
                messagebox.showwarning(message='Not an integer!')
                return

        assert self.frame_idx <= len(self.labels)
        
        if self.frame_idx == len(self.labels):
            self.labels.append(new_label)
        else:
            self.labels[self.frame_idx] = new_label
        self.frame_idx += 1
        if self.frame_idx < len(self.labels):
            if self.labels[self.frame_idx]==NO_LABEL:
                self.radio_var.set(RADIO_NON)
            else:
                self.radio_var.set(RADIO_LABEL)
            self.entry_var.set(str(self.labels[self.frame_idx]))
        else:
            self.entry_var.set('')
        self.update()

    def d_click_event_handler(self, event:tk.Event):
        """
        [Prev] key
        """
        if self.radio_var.get()==RADIO_NON:
            new_label = NO_LABEL
        elif self.radio_var.get()==RADIO_LABEL:
            try:
                new_label = int(self.entry_var.get())
            except ValueError:
                messagebox.showwarning(message='Not an integer!')
                return

        assert self.frame_idx <= len(self.labels)
        
        if self.frame_idx == len(self.labels):
            self.labels.append(new_label)
        else:
            self.labels[self.frame_idx] = new_label
        self.frame_idx = max(0, self.frame_idx-1)
        if self.frame_idx < len(self.labels):
            if self.labels[self.frame_idx]==NO_LABEL:
                self.radio_var.set(RADIO_NON)
            else:
                self.radio_var.set(RADIO_LABEL)
            self.entry_var.set(str(self.labels[self.frame_idx]))
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
        self.index_var.set(str(self.frame_idx)+'/'+str(len(self.labels)))
        self.radio_button_command()

    def run(self):
        self.load_vid()
        self.update()
        self.root.mainloop()
        


if __name__ == '__main__':
    Console().run()