import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2

class Console():
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title('Video Labeler')
        self.mainframe = ttk.Frame(self.root)
        self.mainframe.grid(column=0,row=0)

        self.label_image = ttk.Label(self.mainframe)
        self.label_image.grid(column=0,row=0,columnspan=2,rowspan=3)

        # TODO : Fill commands
        self.button_cut = ttk.Button(
            self.mainframe,
            text='Cut',
            # command=
        )
        self.button_cut.grid(column=0, row=3)
        
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
            # command=
        )
        self.button_open.grid(column=4,row=1)
    
    def load_vid(self):
        self.vid_name = filedialog.askopenfilename()
        print(self.vid_name)
        cap = cv2.VideoCapture(self.vid_name)
        self.frames = []
        while (cap.isOpened()):
            ret,frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame)
            else:
                break

    def run(self):

        self.root.mainloop()


if __name__ == '__main__':
    Console().run()