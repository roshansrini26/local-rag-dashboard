import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

DATA_DIR = "data"
SUPPORTED_EXTENSIONS = (".pdf", ".docx")
DEBOUNCE_SECONDS = 5  # seconds

class NewFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.pending_files = set()
        self.last_event_time = {}

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(SUPPORTED_EXTENSIONS):
            print(f"\n Detected new file: {event.src_path}")
            self.pending_files.add(event.src_path)
            self.last_event_time[event.src_path] = time.time()

    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(SUPPORTED_EXTENSIONS) and event.src_path in self.pending_files:
            self.last_event_time[event.src_path] = time.time()

    def check_and_process(self):
        now = time.time()
        ready_files = [
            f for f in self.pending_files
            if now - self.last_event_time.get(f,0) >= DEBOUNCE_SECONDS
        ]
        if ready_files:
            print(f"\n Running ingestion for {len(ready_files)} new files(s) ..")
            subprocess.run(["python3", "ingest.py"])
            for f in ready_files:
                self.pending_files.discard(f)


def main():
    print(f"Watching '{DATA_DIR}/' for new files")
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, DATA_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(2)
            event_handler.check_and_process()
    except KeyboardInterrupt:
        observer.stop()
        print("\n Watched Stopped.")

    observer.join()


if __name__ == "__main__":
    main()