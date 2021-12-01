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
RADIO_SKIP = 2

NO_LABEL = 'NL'
SKIP_LABEL = 'skip'

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

        self.radio_skip = ttk.Radiobutton(
            self.mainframe,
            variable=self.radio_var,
            value=2,
            command=self.radio_button_command,
        )
        self.radio_skip.grid(column=2,row=2)

        self.label_skip = ttk.Label(
            self.mainframe,
            text='Skip'
        )
        self.label_skip.grid(column=3,row=2)
        


        self.label_info = ttk.Label(
            self.mainframe,
            text='Next: F or Enter\nPrev: D'
        )
        self.label_info.grid(column=3, row=3)

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

        self.button_save = ttk.Button(
            self.mainframe,
            text='save',
            command=self.button_save_command,
        )
        self.button_save.grid(column=4,row=4)

    
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
        if self.radio_var.get() != RADIO_LABEL:
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

    def button_save_command(self):
        save_dir = Path(filedialog.askdirectory())
        non_label_count = 0
        for frame, label in zip(self.frames,self.labels):
            if label == SKIP_LABEL:
                continue
            if self.cut_point_2 is None:
                cut_frame = frame
            else:
                cut_frame = frame[self.cut_point_1[0]:self.cut_point_2[0],
                                  self.cut_point_1[1]:self.cut_point_2[1]]

            cut_frame = cv2.cvtColor(cut_frame,cv2.COLOR_RGB2BGR)
            if label == NO_LABEL:
                new_name = NO_LABEL + '_' + str(non_label_count)
                non_label_count+=1
            else:
                new_name = str(label)
            new_path = str((save_dir/new_name).with_suffix('.png'))
            cv2.imwrite(new_path, cut_frame)

    def image_click_event_handler(self,event:tk.Event):
        x_f = int(max(event.x-5,0)/self.ratio)
        y_f = int(max(event.y-5,0)/self.ratio)
        if self.cut_mode == CUT_WAITING1:
            self.cut_point_1 = (y_f,x_f)
            self.cut_mode = CUT_WAITING2
            self.update()
        elif self.cut_mode == CUT_WAITING2:
            new_point1 = (min(self.cut_point_1[0],y_f),
                          min(self.cut_point_1[1],x_f))
            new_point2 = (max(self.cut_point_1[0],y_f),
                          max(self.cut_point_1[1],x_f))
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
        elif self.radio_var.get()==RADIO_SKIP:
            new_label = SKIP_LABEL

        assert self.frame_idx <= len(self.labels)
        
        if self.frame_idx == len(self.labels):
            self.labels.append(new_label)
        else:
            self.labels[self.frame_idx] = new_label
        self.frame_idx += 1
        if self.frame_idx < len(self.labels):
            if self.labels[self.frame_idx]==NO_LABEL:
                self.radio_var.set(RADIO_NON)
            elif self.labels[self.frame_idx]==SKIP_LABEL:
                self.radio_var.set(RADIO_SKIP)
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
        elif self.radio_var.get()==RADIO_SKIP:
            new_label = SKIP_LABEL

        assert self.frame_idx <= len(self.labels)
        
        if self.frame_idx == len(self.labels):
            self.labels.append(new_label)
        else:
            self.labels[self.frame_idx] = new_label
        self.frame_idx = max(0, self.frame_idx-1)
        if self.frame_idx < len(self.labels):
            if self.labels[self.frame_idx]==NO_LABEL:
                self.radio_var.set(RADIO_NON)
            elif self.labels[self.frame_idx]==SKIP_LABEL:
                self.radio_var.set(RADIO_SKIP)
            else:
                self.radio_var.set(RADIO_LABEL)
            self.entry_var.set(str(self.labels[self.frame_idx]))
        self.update()

    def update(self):
        cur_raw_frame = self.frames[self.frame_idx].copy()
        if self.cut_point_2 is None:
            cut_frame = cur_raw_frame
        else:
            cut_frame = cur_raw_frame[self.cut_point_1[0]:self.cut_point_2[0],
                                      self.cut_point_1[1]:self.cut_point_2[1]]
        if cut_frame.shape[1]>1200:
            self.ratio = 1200/cut_frame.shape[1]
            cut_frame=cv2.resize(cut_frame, (0,0),fx=self.ratio, fy=self.ratio)
        else:
            self.ratio = 1
        self.current_image = ImageTk.PhotoImage(Image.fromarray(
            cut_frame
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