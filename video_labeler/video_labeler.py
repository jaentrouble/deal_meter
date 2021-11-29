import tkinter as tk
from tkinter import ttk
from PIL import ImageTk, Image

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
            text='Not-labelable',
            variable=self.radio_var,
            value=0,
        )
        self.radio_nonlabel.grid(column=2,row=0)
        
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
        