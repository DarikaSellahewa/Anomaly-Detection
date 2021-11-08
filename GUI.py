# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:53:32 2021

@author: darik
"""

import tkinter as tk


window = tk.Tk()

frame = tk.Frame()
label = tk.Label(master = frame, text="Hello, Tkinter")
entry = tk.Entry(master = frame)
label_t = tk.Label(master = frame)

label_t["text"] = entry.get()

label.pack()
entry.pack()
frame.pack()
label_t.pack()
window.mainloop()
