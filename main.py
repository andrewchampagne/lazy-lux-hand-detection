import tkinter as tk
import subprocess
import os
import time


process = None

def start_script():
    global process
    if process is None or process.poll() is not None:
        script_path = "./inference_classifier.py"  
        process = subprocess.Popen(["python", script_path], shell=True)
        while not os.path.exists("status.txt"):
            time.sleep(0.1)
        start_button.config(text="Loaded!")

def stop_script():
    global process
    if process is not None and process.poll() is None:
        with open("stop_flag.txt", "w") as f:
            f.write("stop")
        # Terminate the external script
        process.terminate()  # Attempt to terminate gracefully
        try:
            process.wait(timeout=5)  # Wait for process to terminate
        except subprocess.TimeoutExpired:
            process.kill()  # Force kill if terminate failsq
        start_button.config(text="Start Lazy Lux")
        print("process terminated")
        process = None
        
def on_closing():
    stop_script()  
    window.destroy() 

window = tk.Tk()
window.title("Lazy Lux")
window.geometry("250x100")


start_button = tk.Button(window, text="Start Lazy Lux", command=start_script)
start_button.pack(pady=10)


stop_button = tk.Button(window, text="Stop Lazy Lux", command=stop_script)
stop_button.pack(pady=10)

window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()