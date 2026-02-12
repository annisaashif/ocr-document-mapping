import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List

# ============================
# HUBUNGKAN KE PARSER KAMU
# ============================
try:
    from app_proses_pdf import process_one_pdf
    print("IMPORT OK: process_one_pdf from app_proses_pdf")
except Exception as e:
    process_one_pdf = None
    print("IMPORT FAILED app_proses_pdf:", e)



def process_pdf_file(pdf_path: str, mysql_server: str, db_name: str, debug: bool = False):
    if process_one_pdf is None:
        raise RuntimeError("Parser belum terhubung. Pastikan app_proses_pdf.py punya fungsi process_one_pdf() dan tidak error saat import.")
    return process_one_pdf(pdf_path, mysql_server, db_name, debug=debug)



class PdfRow:
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self.var = tk.BooleanVar(value=False)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF → MySQL Loader (Tkinter)")
        self.geometry("1200x700")
        self.minsize(1100, 650)

        self.pdf_folder_var = tk.StringVar()
        self.mysql_server_var = tk.StringVar(value="mysql+pymysql://root:123456@192.168.10.40:3306")
        self.db_name_var = tk.StringVar(value="tds_perfile_db")
        self.search_var = tk.StringVar()

        self.status_var = tk.StringVar(value="Ready.")
        self.rows: List[PdfRow] = []
        self.filtered_rows: List[PdfRow] = []

        self.msg_q = queue.Queue()
        self.worker_thread = None

        self.build_ui()
        self.search_var.trace_add("write", lambda *_: self.apply_filter())
        self.after(100, self.poll_queue)

    def build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        # ---- Top: folder picker ----
        top = ttk.Frame(root)
        top.pack(fill="x")

        ttk.Label(top, text="PDF Folder:").pack(side="left")
        self.pdf_entry = ttk.Entry(top, textvariable=self.pdf_folder_var, width=70)
        self.pdf_entry.pack(side="left", padx=8, fill="x", expand=True)
        ttk.Button(top, text="Browse", command=self.browse_folder).pack(side="left")

        # ---- DB config ----
        db = ttk.LabelFrame(root, text="Database", padding=10)
        db.pack(fill="x", pady=10)

        db_row = ttk.Frame(db)
        db_row.pack(fill="x")
        ttk.Label(db_row, text="MySQL Server URL:").pack(side="left")
        ttk.Entry(db_row, textvariable=self.mysql_server_var, width=55).pack(side="left", padx=8)
        ttk.Label(db_row, text="DB Name:").pack(side="left", padx=(15, 0))
        ttk.Entry(db_row, textvariable=self.db_name_var, width=25).pack(side="left", padx=8)

        # ---- Load + status ----
        loadbar = ttk.Frame(root)
        loadbar.pack(fill="x", pady=6)

        self.btn_load = ttk.Button(loadbar, text="Load Data", command=self.load_data)
        self.btn_load.pack(side="left")

        self.progress = ttk.Progressbar(loadbar, orient="horizontal", mode="determinate", length=320)
        self.progress.pack(side="left", padx=10)
        self.progress["maximum"] = 100
        self.progress["value"] = 0

        ttk.Label(loadbar, textvariable=self.status_var).pack(side="left", padx=10)

        # ---- Controls (select/search) ----
        ctrl = ttk.Frame(root)
        ctrl.pack(fill="x", pady=6)

        ttk.Button(ctrl, text="Select All", command=self.select_all).pack(side="left")
        ttk.Button(ctrl, text="Deselect All", command=self.deselect_all).pack(side="left", padx=6)

        ttk.Label(ctrl, text="Search:").pack(side="left", padx=(20, 4))
        self.search_entry = ttk.Entry(ctrl, textvariable=self.search_var, width=45)
        self.search_entry.pack(side="left")

        # =========================================================
        # MAIN AREA (FIX: PDF & LOG BOX HEIGHT MUST MATCH)
        # =========================================================
        BORDER_COLOR = "#bdbdbd"
        BORDER_THICKNESS = 1
        BOX_HEIGHT = 300      # <<< ubah ini untuk pendekin/tinggikan kotak
        RIGHT_WIDTH = 500     # <<< lebar log (kanan). PDF otomatis lebih lebar.

        main = ttk.Frame(root)
        main.pack(fill="x", pady=10)   # tidak expand vertikal

        # LEFT (PDF) - lebar fleksibel
        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        ttk.Label(left, text="PDF Files").pack(anchor="w")

        # kotak pdf FIX height
        pdf_box = tk.Frame(left, height=BOX_HEIGHT)
        pdf_box.pack(fill="x")
        pdf_box.pack_propagate(False)

        self.pdf_border = tk.Canvas(
            pdf_box,
            highlightthickness=BORDER_THICKNESS,
            highlightbackground=BORDER_COLOR,
            bd=0,
            bg="white"
        )
        self.pdf_border.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.pdf_border, borderwidth=0, highlightthickness=0, bg="white")
        self.scrollbar = ttk.Scrollbar(self.pdf_border, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.list_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        self.list_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # scroll wheel
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)

        self.btn_process = ttk.Button(left, text="Process Data", command=self.process_selected)
        self.btn_process.pack(fill="x", pady=(10, 0))

        # RIGHT (LOG) - lebar tetap lebih kecil
        right = ttk.Frame(main, width=RIGHT_WIDTH)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)  # penting: biar width kanan tetap

        ttk.Label(right, text="Log").pack(anchor="w")

        # kotak log FIX height (SAMA persis)
        log_box = tk.Frame(right, height=BOX_HEIGHT)
        log_box.pack(fill="x")
        log_box.pack_propagate(False)

        self.log_border = tk.Canvas(
            log_box,
            highlightthickness=BORDER_THICKNESS,
            highlightbackground=BORDER_COLOR,
            bd=0,
            bg="white"
        )
        self.log_border.pack(fill="both", expand=True)

        self.log = tk.Text(self.log_border, wrap="word", borderwidth=0, highlightthickness=0)
        self.log_scrollbar = ttk.Scrollbar(self.log_border, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=self.log_scrollbar.set)
        self.log_scrollbar.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)

        self.btn_clear_log = ttk.Button(right, text="Clear Log", command=lambda: self.log.delete("1.0", "end"))
        self.btn_clear_log.pack(fill="x", pady=(10, 0))

    # ---------------- UI helpers ----------------
    def on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def log_msg(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self.btn_load.config(state=state)
        self.btn_process.config(state=state)

    def browse_folder(self):
        path = filedialog.askdirectory(title="Select PDF Folder")
        if path:
            self.pdf_folder_var.set(path)

    def clear_list(self):
        for child in self.list_frame.winfo_children():
            child.destroy()

    def render_list(self):
        self.clear_list()
        for r in self.filtered_rows:
            cb = ttk.Checkbutton(self.list_frame, text=r.name, variable=r.var)
            cb.pack(anchor="w", pady=2, padx=6)

    def apply_filter(self):
        q = self.search_var.get().strip().lower()
        if not q:
            self.filtered_rows = list(self.rows)
        else:
            self.filtered_rows = [r for r in self.rows if q in r.name.lower()]
        self.render_list()

    def select_all(self):
        for r in self.filtered_rows:
            r.var.set(True)

    def deselect_all(self):
        for r in self.filtered_rows:
            r.var.set(False)

    # ---------------- Load Data ----------------
    def load_data(self):
        folder = self.pdf_folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Folder PDF tidak valid.")
            return

        self.rows = []
        self.filtered_rows = []
        self.clear_list()
        self.progress["value"] = 0
        self.status_var.set("Scanning...")
        self.log_msg(f"Loading PDFs from: {folder}")
        self.set_busy(True)

        def worker():
            try:
                pdfs = []
                scanned = 0
                for root, _, files in os.walk(folder):
                    for fn in files:
                        scanned += 1
                        if fn.lower().endswith(".pdf"):
                            pdfs.append(os.path.join(root, fn))
                        if scanned % 500 == 0:
                            self.msg_q.put(("log", f"Load progress: scanned {scanned} files... found PDF={len(pdfs)}"))
                pdfs.sort()
                self.msg_q.put(("loaded", pdfs))
            except Exception as e:
                self.msg_q.put(("error", str(e)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def on_loaded(self, pdfs: List[str]):
        self.rows = [PdfRow(p) for p in pdfs]
        self.filtered_rows = list(self.rows)
        self.render_list()
        self.status_var.set(f"Loaded {len(pdfs)} PDFs.")
        self.log_msg(f"Load finished: total PDF found = {len(pdfs)}")
        self.set_busy(False)

    # ---------------- Process Data ----------------
    def process_selected(self):
        mysql_server = self.mysql_server_var.get().strip()
        db_name = self.db_name_var.get().strip()
        if not mysql_server or not db_name:
            messagebox.showerror("Error", "MySQL Server URL dan DB Name wajib diisi.")
            return

        selected = [r.path for r in self.rows if r.var.get()]
        if not selected:
            self.log_msg("Tidak ada PDF yang dipilih.")
            return

        self.set_busy(True)
        self.progress["value"] = 0
        self.progress["maximum"] = len(selected)
        self.status_var.set("Processing...")
        self.log_msg(f"Start processing {len(selected)} PDFs...")

        def worker():
            try:
                total = len(selected)
                done = 0
                for p in selected:
                    try:
                        table_name = process_pdf_file(p, mysql_server, db_name, debug=False)
                        done += 1
                        if done % 10 == 0 or done == total:
                            self.msg_q.put(("log", f"[OK] {done}/{total} -> {os.path.basename(p)} (table={table_name})"))
                        self.msg_q.put(("progress", done, total))
                    except Exception as e:
                        done += 1
                        self.msg_q.put(("log", f"[FAIL] {done}/{total} -> {os.path.basename(p)} : {e}"))
                        self.msg_q.put(("progress", done, total))
                self.msg_q.put(("done", total))
            except Exception as e:
                self.msg_q.put(("error", str(e)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    # ---------------- Queue polling ----------------
    def poll_queue(self):
        try:
            while True:
                msg = self.msg_q.get_nowait()
                kind = msg[0]

                if kind == "log":
                    self.log_msg(msg[1])
                elif kind == "loaded":
                    self.on_loaded(msg[1])
                elif kind == "progress":
                    done, total = msg[1], msg[2]
                    self.progress["value"] = done
                    self.status_var.set(f"Processing {done}/{total}...")
                elif kind == "done":
                    total = msg[1]
                    self.status_var.set(f"Done. Processed {total} PDFs.")
                    self.log_msg("Process finished.")
                    self.set_busy(False)
                elif kind == "error":
                    self.set_busy(False)
                    self.status_var.set("Error.")
                    self.log_msg(f"ERROR: {msg[1]}")
                    messagebox.showerror("Error", msg[1])

        except queue.Empty:
            pass

        self.after(100, self.poll_queue)


if __name__ == "__main__":
    App().mainloop()
