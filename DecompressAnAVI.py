import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import subprocess

def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path

def get_output_file():
    while True:
        root = tk.Tk()
        root.withdraw()
        output_file = filedialog.asksaveasfilename(
            title='Save Decompressed Video',
            defaultextension='.avi',
            filetypes=[('AVI files', '*.avi'), ('MP4 files', '*.mp4')]
        )
        root.destroy()

        if not output_file:
            return None  # User cancelled

        base, ext = os.path.splitext(output_file)
        if ext.lower() not in ['.avi', '.mp4']:
            ext = '.avi'  

        output_file = base + ext

        if os.path.exists(output_file):
            if messagebox.askyesno("File exists", f"{output_file} already exists. Overwrite?"):
                return output_file
        else:
            return output_file

def decompress_video_ffmpeg(input_path, output_path, grayscale=False):
    """
    Uses FFmpeg to convert a compressed video to an uncompressed or losslessly compressed AVI.
    - grayscale: if True, output will be single-channel (grayscale).
    """
    if grayscale:
        pix_fmt = 'gray'
    else:
        pix_fmt = 'yuv420p'  # or another appropriate format

    # Choose codec: rawvideo for uncompressed, ffv1 for lossless, mjpeg for compatibility
    codec = 'rawvideo'  # Change to 'ffv1' or 'mjpeg' if preferred

    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', codec,
        '-pix_fmt', pix_fmt,
        output_path
    ]

    # Create a small TK window to show progress
    root = tk.Tk()
    root.title("Decompressing Video with FFmpeg")
    root.geometry("300x100")

    progress_label = tk.Label(root, text="Decompressing video using FFmpeg...")
    progress_label.pack(pady=10)

    progress_bar = Progressbar(root, length=200, mode='indeterminate')
    progress_bar.pack(pady=10)
    progress_bar.start()

    try:
        # Run FFmpeg command
        subprocess.run(command, check=True)
        progress_label.config(text="Decompression completed!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred during video decompression: {str(e)}")
    finally:
        progress_bar.stop()
        root.destroy()

def main():
    try:
        input_file = select_file("Select Compressed Video", [("Video Files", "*.mp4;*.avi;*.mov")])
        if not input_file:
            messagebox.showerror("Error", "No input file selected.")
            return

        output_file = get_output_file()
        if not output_file:
            messagebox.showerror("Error", "No output file specified.")
            return

        # Ask if user wants grayscale or color
        msg = "Convert video to grayscale? (Recommended for tracking tools like Ctrax)"
        convert_to_gray = messagebox.askyesno("Color or Grayscale?", msg)

        decompress_video_ffmpeg(input_file, output_file, grayscale=convert_to_gray)
        messagebox.showinfo("Success", f"Decompression completed! Saved as {output_file}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
