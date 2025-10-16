import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image
import threading

def select_directory():
    directory = filedialog.askdirectory()
    if directory:
        threading.Thread(target=save_multipage_tiff, args=(directory,)).start()

def extract_numeric_part(filename):
    numeric_part = re.findall(r'\d+', filename)
    return int(numeric_part[0]) if numeric_part else float('inf')

def save_multipage_tiff(directory):
    tiff_files = [f for f in os.listdir(directory) if f.lower().endswith('.tif')]
    tiff_files.sort(key=extract_numeric_part)

    if not tiff_files:
        messagebox.showerror("Error", "No TIFF files found in the selected directory.")
        return

    images = []
    total_files = len(tiff_files)

    progress['maximum'] = total_files
    log_file = os.path.join(directory, "processing_log.txt")
    
    with open(log_file, 'w') as log:
        log.write("Processing started.\n")
        for i, tiff_file in enumerate(tiff_files):
            img_path = os.path.join(directory, tiff_file)
            try:
                img = Image.open(img_path)
                img.verify()
                img = Image.open(img_path)
                img = img.convert("RGB")
                if images:
                    img = img.resize(images[0].size)
                images.append(img)
                log.write(f"Processed file: {tiff_file}\n")
            except Exception as e:
                log.write(f"Skipping file {tiff_file}: {e}\n")
            
            progress['value'] = i + 1
            root.update_idletasks()

    if images:
        save_path = os.path.join(directory, "multipage_tiff.tif")
        try:
            images[0].save(save_path, save_all=True, append_images=images[1:], compression="tiff_deflate")
            messagebox.showinfo("Success", f"Multipage TIFF saved as {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save multipage TIFF: {e}")
    else:
        messagebox.showerror("Error", "No valid images to save.")

    with open(log_file, 'a') as log:
        log.write("Processing completed.\n")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    root.deiconify()
    root.title("TIFF to Multi-Page TIFF Converter")

    tk.Label(root, text="Processing...").grid(row=0, column=0, padx=10, pady=10)
    progress = ttk.Progressbar(root, orient='horizontal', length=300, mode='determinate')
    progress.grid(row=1, column=0, padx=10, pady=10)
    
    select_directory()

    root.mainloop()
