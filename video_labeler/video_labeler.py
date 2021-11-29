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
        self.label_image.grid(column=0,row=0,columnspan=2)

        # TODO : Fill commands
        self.button_cut = ttk.Button(
            self.mainframe,
            text='Cut',
            # command=
        )
        self.button_cut.grid(column=0, row=1)
        
        self.button_reset = ttk.Button(
            self.mainframe,
            text='Reset',
            # command=
        )
        self.button_reset.grid(column=1)
        