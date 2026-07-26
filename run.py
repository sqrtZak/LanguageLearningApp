import sys
import sqlite3
import os
import datetime
import concurrent.futures
import multiprocessing
import uuid
import string
import subprocess
import tempfile
import hashlib
from difflib import SequenceMatcher
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QListWidget, 
                             QLabel, QFileDialog, QMessageBox, QProgressBar, QTextEdit,
                             QListWidgetItem)
from PyQt6.QtCore import pyqtSlot, QThread, pyqtSignal, Qt
from PyQt6.QtGui import QAction, QCloseEvent

import genanki

# ==========================================
# 1. Top-Level Parallel Worker Function (Robust Global Alignment)
# ==========================================
# ==========================================
# 1. Top-Level Parallel Worker Function (Robust Global Alignment)
# ==========================================
def process_single_song(task):
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    import whisperx
    import os
    import uuid
    import string
    import subprocess
    import tempfile
    from difflib import SequenceMatcher

    audio_path = task['audio']
    lyrics_path = task['lyrics']
    title = os.path.basename(audio_path)
    
    device = "cpu"
    compute_type = "int8"
    
    # Initialize temp paths so we can guarantee cleanup
    vocals_path = None
    stripped_wav_path = None
    
    try:
        # --- THE FIX: Strip the audio into a "regular" WAV file ---
        temp_dir = tempfile.gettempdir()
        stripped_wav_path = os.path.join(temp_dir, f"stripped_audio_{uuid.uuid4().hex}.wav")
        vocals_path = os.path.join(temp_dir, f"temp_vocals_{uuid.uuid4().hex}.wav")

        # FFmpeg reads the MP3, strips all ID3 tags/album art, and outputs a raw WAV
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-vn",          # Strip any video/album art
            "-ar", "44100", # Standardize sample rate
            "-ac", "2",     # Standardize to stereo
            stripped_wav_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Load isolation and transcription models
        demucs_model = get_model("htdemucs")
        demucs_model.to(device)
        demucs_model.eval()
        
        whisper_model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        align_model, align_metadata = whisperx.load_align_model(language_code="ru", device=device)
        
        # Audio preprocessing (Load the stripped WAV, not the original MP3)
        wav, sr = torchaudio.load(stripped_wav_path)
        
        if sr != demucs_model.samplerate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=demucs_model.samplerate)
            wav = resampler(wav)
            sr = demucs_model.samplerate
            
        wav = wav.to(device)
        with torch.no_grad():
            sources = apply_model(demucs_model, wav.unsqueeze(0), split=True)[0]
            
        vocals_idx = demucs_model.sources.index("vocals")
        vocals_tensor = sources[vocals_idx].cpu()
        
        torchaudio.save(vocals_path, vocals_tensor, sr)
        
        # Audio Transcription & Word-Level Alignment
        audio = whisperx.load_audio(vocals_path)
        transcribe_result = whisper_model.transcribe(audio, batch_size=8, language="ru")
        
        aligned_result = whisperx.align(
            transcribe_result["segments"], align_model, align_metadata, audio, device, return_char_alignments=False
        )

        # Parse Reference Lyrics File
        with open(lyrics_path, 'r', encoding='utf-8') as f:
            raw_lines = [line.strip() for line in f.readlines() if line.strip()]

        user_ru = raw_lines[0::2]
        user_en = raw_lines[1::2]

        # Flatten Reference Lyrics for Structural Mapping
        user_words = []
        word_to_line_map = []
        for line_idx, ru_line in enumerate(user_ru):
            clean_line = ru_line.translate(str.maketrans('', '', string.punctuation)).lower().split()
            for word in clean_line:
                user_words.append(word)
                word_to_line_map.append(line_idx)

        if not user_words:
            return {"success": False, "title": title, "error": "Reference lyrics file is empty."}

        # Flatten Model Transcribed Words
        whisper_word_objects = []
        whisper_clean_words = []
        for segment in aligned_result["segments"]:
            if "words" in segment:
                for word in segment["words"]:
                    if "start" in word and "end" in word:
                        clean_text = word["word"].translate(str.maketrans('', '', string.punctuation)).lower().strip()
                        if clean_text:
                            whisper_word_objects.append(word)
                            whisper_clean_words.append(clean_text)

        total_duration = whisper_word_objects[-1]["end"] if whisper_word_objects else 180.0

        # Global Structural Matching Block Detection
        matcher = SequenceMatcher(None, whisper_clean_words, user_words)
        matching_blocks = matcher.get_matching_blocks()

        user_word_times = [None] * len(user_words)
        for block in matching_blocks:
            for i in range(block.size):
                w_idx = block.a + i
                u_idx = block.b + i
                user_word_times[u_idx] = whisper_word_objects[w_idx]["start"]

        # Linear Interpolation Fill-in Strategy for Missing/Mumbled Words
        known_indices = [i for i, t in enumerate(user_word_times) if t is not None]

        if not known_indices:
            for i in range(len(user_words)):
                user_word_times[i] = (i / len(user_words)) * total_duration
        else:
            first_idx = known_indices[0]
            first_time = user_word_times[first_idx]
            for i in range(first_idx):
                user_word_times[i] = (i / first_idx) * first_time if first_idx > 0 else first_time
                
            last_idx = known_indices[-1]
            last_time = user_word_times[last_idx]
            for i in range(last_idx + 1, len(user_words)):
                remaining = len(user_words) - 1 - last_idx
                passed = i - last_idx
                user_word_times[i] = last_time + (passed / remaining) * (total_duration - last_time)

            for k in range(len(known_indices) - 1):
                idx1, idx2 = known_indices[k], known_indices[k+1]
                t1, t2 = user_word_times[idx1], user_word_times[idx2]
                for i in range(idx1 + 1, idx2):
                    fraction = (i - idx1) / (idx2 - idx1)
                    user_word_times[i] = t1 + fraction * (t2 - t1)

        # Segment Construction
        final_segments = []
        for line_idx in range(len(user_ru)):
            line_word_indices = [i for i, l in enumerate(word_to_line_map) if l == line_idx]
            if not line_word_indices:
                continue
            
            start_word_idx = line_word_indices[0]
            end_word_idx = line_word_indices[-1]
            
            start_sec = user_word_times[start_word_idx]
            if end_word_idx + 1 < len(user_word_times):
                end_sec = user_word_times[end_word_idx + 1]
            else:
                end_sec = start_sec + 2.5
                
            if end_sec <= start_sec:
                end_sec = start_sec + 2.0
            if (end_sec - start_sec) > 15.0:
                end_sec = start_sec + 4.0
                
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            
            translation_en = user_en[line_idx] if line_idx < len(user_en) else ""
            final_segments.append((start_ms, end_ms, user_ru[line_idx], translation_en))

        return {"success": True, "title": title, "audio_path": audio_path, "segments": final_segments}

    except Exception as e:
        import traceback
        return {"success": False, "title": title, "error": traceback.format_exc()}
        
    finally:
        # Guarantee that temporary audio files are deleted to prevent storage leaks
        if vocals_path and os.path.exists(vocals_path):
            try:
                os.remove(vocals_path)
            except Exception:
                pass
        if stripped_wav_path and os.path.exists(stripped_wav_path):
            try:
                os.remove(stripped_wav_path)
            except Exception:
                pass

class ParallelAIWorker(QThread):
    progress = pyqtSignal(str)          
    progress_percent = pyqtSignal(int)  
    log_message = pyqtSignal(str)       
    song_finished = pyqtSignal(str, str, list) 
    batch_finished = pyqtSignal()

    def __init__(self, task_queue, max_workers=2):
        super().__init__()
        self.task_queue = task_queue 
        self.max_workers = max_workers

    def run(self):
        self.log_message.emit(f"🚀 Starting Parallel Pipeline with {self.max_workers} concurrent workers...")
        self.progress_percent.emit(5)
        
        completed_tasks = 0
        total_tasks = len(self.task_queue)

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {executor.submit(process_single_song, task): task for task in self.task_queue}
            
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    if result["success"]:
                        self.log_message.emit(f" Finished Processing: {result['title']}")
                        self.song_finished.emit(result["title"], result["audio_path"], result["segments"])
                    else:
                        self.log_message.emit(f" Error processing {result['title']}:\n{result['error']}")
                except Exception as e:
                    self.log_message.emit(f" Fatal Worker Error: {str(e)}")
                
                completed_tasks += 1
                percent = int(5 + (95 * (completed_tasks / total_tasks)))
                self.progress_percent.emit(percent)
                self.progress.emit(f"Completed {completed_tasks} of {total_tasks} songs")

        self.progress_percent.emit(100)
        self.log_message.emit("\n Entire parallel batch finished successfully!")
        self.batch_finished.emit()


class AnkiExportWorker(QThread):
    progress_percent = pyqtSignal(int)
    finished = pyqtSignal(str, bool)

    def __init__(self, song_id, db_path, export_dir):
        super().__init__()
        self.song_id = song_id
        self.db_path = db_path
        self.export_dir = export_dir

    def get_stable_id(self, string_val):
        return int(hashlib.sha256(string_val.encode('utf-8')).hexdigest(), 16) % (1 << 31)

    def run(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT title, audio_path FROM songs WHERE id = ?", (self.song_id,))
            song = cursor.fetchone()
            if not song:
                self.finished.emit("Song not found in DB.", False)
                return
            
            title, audio_path = song
            cursor.execute("SELECT id, start_ts, end_ts, lyric_ru, translation_en FROM segments WHERE song_id = ? ORDER BY start_ts ASC", (self.song_id,))
            segments = cursor.fetchall()
            conn.close()

            safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c==' ']).rstrip().replace(" ", "_")
            deck_name = f"KinoLingo::{title.replace('.mp3', '').replace('.wav', '')}"
            
            deck_id = self.get_stable_id(deck_name)
            model_id = self.get_stable_id("KinoLingo_Model_v1")

            my_model = genanki.Model(
                model_id,
                'KinoLingo Audio Model',
                fields=[{'name': 'Audio'}, {'name': 'Russian'}, {'name': 'English'}],
                templates=[{
                    'name': 'Listening Card',
                    'qfmt': '<div style="text-align: center; margin-top: 20px;">{{Audio}}</div>',
                    'afmt': '{{FrontSide}}<hr id="answer"><div style="text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 10px;">{{Russian}}</div><div style="text-align: center; font-size: 18px; color: #555;">{{English}}</div>',
                }],
                css='.card { font-family: arial; font-size: 20px; text-align: center; color: black; background-color: white; }'
            )

            my_deck = genanki.Deck(deck_id, deck_name)
            media_files = []

            with tempfile.TemporaryDirectory() as temp_dir:
                total = len(segments)
                
                for i, seg in enumerate(segments):
                    seg_id, start_ts, end_ts, ru, en = seg
                    
                    start_sec = start_ts / 1000.0
                    end_sec = end_ts / 1000.0
                    audio_filename = f"kinolingo_{safe_title}_{seg_id}.mp3"
                    out_audio = os.path.join(temp_dir, audio_filename)
                    
                    cmd = [
                        "ffmpeg", "-y", "-i", audio_path,
                        "-ss", str(start_sec), "-to", str(end_sec),
                        "-q:a", "0", "-map", "a", out_audio
                    ]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    media_files.append(out_audio)
                    
                    note = genanki.Note(
                        model=my_model,
                        fields=[f"[sound:{audio_filename}]", ru, en]
                    )
                    my_deck.add_note(note)
                    self.progress_percent.emit(int(((i + 1) / total) * 100))

                package = genanki.Package(my_deck)
                package.media_files = media_files
                
                output_apkg_path = os.path.join(self.export_dir, f"{safe_title}.apkg")
                package.write_to_file(output_apkg_path)

            success_msg = (f"Successfully created Anki Deck!\n\nSaved To:\n{output_apkg_path}")
            self.finished.emit(success_msg, True)
            
        except Exception as e:
            import traceback
            self.finished.emit(f"Error: {str(e)}\n{traceback.format_exc()}", False)


class LanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KinoLingo - The Russian Anki Deck Factory")
        self.resize(900, 750)

        self.db_path = "learning_data.db"
        self.init_database()

        self.worker = None 
        self.anki_worker = None
        self.import_queue = [] 

        self.setup_ui()
        self.setup_menu() 
        
    def setup_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        action_exit = QAction("Exit", self)
        action_exit.setShortcut("Ctrl+Q")
        action_exit.triggered.connect(self.close)
        file_menu.addAction(action_exit)

    def closeEvent(self, event: QCloseEvent):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, 'Warning', "The AI is processing a batch. Quit and cancel?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.worker.terminate() 
                self.worker.wait(2000)  
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS songs (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, audio_path TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS segments (id INTEGER PRIMARY KEY AUTOINCREMENT, song_id INTEGER, start_ts INTEGER, end_ts INTEGER, lyric_ru TEXT, translation_en TEXT, FOREIGN KEY(song_id) REFERENCES songs(id))''')
        conn.commit()
        conn.close()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        main_layout.addWidget(QLabel("<b> Your Processed Songs (Ready for Anki)</b>"))
        
        self.song_list = QListWidget()
        main_layout.addWidget(self.song_list)
        
        self.btn_export_anki = QPushButton("Export Selected Song to .APKG")
        self.btn_export_anki.setStyleSheet("background-color: #2b78e4; color: white; font-weight: bold; padding: 10px;")
        self.btn_export_anki.clicked.connect(self.export_to_anki)
        main_layout.addWidget(self.btn_export_anki)

        main_layout.addSpacing(20)

        main_layout.addWidget(QLabel("<b> Batch AI Processor Pipeline</b>"))
        main_layout.addWidget(QLabel("<i>Ensure pairs share the same name (e.g. 'Song1.mp3' and 'Song1.txt').</i>"))
        
        queue_controls = QHBoxLayout()
        self.btn_scan_folder = QPushButton("1. Scan Folder for Song Pairs")
        self.btn_scan_folder.clicked.connect(self.scan_folder)
        
        self.btn_process_queue = QPushButton("2. Run AI Pipeline")
        self.btn_process_queue.setEnabled(False)
        self.btn_process_queue.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_process_queue.clicked.connect(self.start_batch_processing)
        
        queue_controls.addWidget(self.btn_scan_folder)
        queue_controls.addWidget(self.btn_process_queue)
        main_layout.addLayout(queue_controls)

        self.queue_ui_list = QListWidget()
        main_layout.addWidget(self.queue_ui_list)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.lbl_import_status = QLabel("Ready")
        main_layout.addWidget(self.lbl_import_status)

        main_layout.addWidget(QLabel("<b>Terminal Output:</b>"))
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;")
        main_layout.addWidget(self.log_viewer)

        self.load_song_list()

    def scan_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing Audio & Lyrics")
        if not folder_path: return

        found_pairs = 0
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.mp3', '.wav', '.flac')):
                base_name = os.path.splitext(file)[0]
                audio_path = os.path.join(folder_path, file)
                lyrics_path = os.path.join(folder_path, f"{base_name}.txt")

                if os.path.exists(lyrics_path):
                    self.import_queue.append({"audio": audio_path, "lyrics": lyrics_path})
                    self.queue_ui_list.addItem(f"⏳ Queued: {base_name}")
                    found_pairs += 1

        if found_pairs > 0:
            self.btn_process_queue.setEnabled(True)
            self.lbl_import_status.setText(f"Found {found_pairs} matching audio/text pairs ready for processing!")
            self.log_viewer.append(f"Successfully scanned folder. Queued {found_pairs} songs.")
        else:
            QMessageBox.information(self, "No Matches Found", "Could not find matching audio and .txt files.")

    def start_batch_processing(self):
        if not self.import_queue: return
        
        self.btn_scan_folder.setEnabled(False)
        self.btn_process_queue.setEnabled(False)
        self.btn_export_anki.setEnabled(False)
        
        self.progress_bar.setValue(0)
        self.log_viewer.clear()
        
        self.worker = ParallelAIWorker(self.import_queue, max_workers=2)
        self.worker.progress.connect(lambda msg: self.lbl_import_status.setText(msg))
        self.worker.progress_percent.connect(self.progress_bar.setValue)
        self.worker.log_message.connect(self.log_viewer.append)
        self.worker.song_finished.connect(self.save_single_song_to_db)
        self.worker.batch_finished.connect(self.cleanup_batch_ui)
        self.worker.start()

    @pyqtSlot(str, str, list)
    def save_single_song_to_db(self, title, audio_path, segments_data):
        if not segments_data: return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO songs (title, audio_path) VALUES (?, ?)", (title, audio_path))
        song_id = cursor.lastrowid
        
        for start_ts, end_ts, lyric, trans in segments_data:
            cursor.execute('''INSERT INTO segments (song_id, start_ts, end_ts, lyric_ru, translation_en) 
                              VALUES (?, ?, ?, ?, ?)''', (song_id, start_ts, end_ts, lyric, trans))
        
        conn.commit()
        conn.close()
        self.load_song_list()
        
        for i in range(self.queue_ui_list.count()):
            item = self.queue_ui_list.item(i)
            if title in item.text() and "⏳" in item.text():
                item.setText(f" Done: {title}")
                break

    @pyqtSlot()
    def cleanup_batch_ui(self):
        self.lbl_import_status.setText("Parallel Processing Complete!")
        self.btn_scan_folder.setEnabled(True)
        self.btn_export_anki.setEnabled(True)
        self.import_queue.clear()

    def load_song_list(self):
        self.song_list.clear()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title FROM songs")
        for row in cursor.fetchall():
            item = QListWidgetItem(row[1])
            item.setData(Qt.ItemDataRole.UserRole, row[0])
            self.song_list.addItem(item)
        conn.close()

    def export_to_anki(self):
        selected_items = self.song_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select a processed song from the list first.")
            return
            
        song_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
        export_dir = QFileDialog.getExistingDirectory(self, "Select Folder to Save Anki Deck")
        if not export_dir: return
            
        self.btn_export_anki.setEnabled(False)
        self.lbl_import_status.setText("Slicing audio and building .APKG file...")
        self.progress_bar.setValue(0)
        
        self.anki_worker = AnkiExportWorker(song_id, self.db_path, export_dir)
        self.anki_worker.progress_percent.connect(self.progress_bar.setValue)
        self.anki_worker.finished.connect(self.on_anki_export_finished)
        self.anki_worker.start()

    @pyqtSlot(str, bool)
    def on_anki_export_finished(self, message, success):
        self.btn_export_anki.setEnabled(True)
        self.lbl_import_status.setText("Ready")
        self.progress_bar.setValue(100)
        if success:
            QMessageBox.information(self, "Export Complete", message)
        else:
            QMessageBox.critical(self, "Export Failed", message)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    import traceback
    try:
        app = QApplication(sys.argv)
        window = LanguageApp()
        window.show()
        sys.exit(app.exec())
    except Exception as run_error:
        print("--- RUNTIME ERROR BLOCK EXECUTED ---")
        traceback.print_exc()
        input("\nPress Enter to exit terminal...")
