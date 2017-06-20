import threading
import time
import tkinter as tk
root = tk.Tk()
root.withdraw()

class Worker(threading.Thread):
    def run(self):
        # long process goes here
        time.sleep(10)

w = Worker()
w.start()
tkMessageBox.showinfo("Work Started", "OK started working")
root.update()
w.join()
tkMessageBox.showinfo("Work Complete", "OK Done")