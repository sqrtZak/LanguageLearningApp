import sys
import sqlite3
import os
import time
import datetime
from difflib import SequenceMatcher
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QStackedWidget, QListWidget, 
                             QLabel, QLineEdit, QFileDialog, QMessageBox)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl, pyqtSlot, QThread, pyqtSignal
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import whisperx
import traceback


# ==========================================
# 1. Background Worker for AI Pipeline
# ==========================================
class AIAlignmentWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(list)

    def __init__(self, audio_path, lyrics_path):
        super().__init__()
        self.audio_path = audio_path
        self.lyrics_path = lyrics_path

    def run(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            # --- Step 1: Vocal Separation with Native Demucs ---
            self.progress.emit("Loading Demucs model...")
            # FIXED: Using get_model directly instead of demucs.api
            demucs_model = get_model("htdemucs")
            demucs_model.to(device)
            demucs_model.eval()

            self.progress.emit("Loading track into memory...")
            wav, sr = torchaudio.load(self.audio_path)
            
            if sr != demucs_model.samplerate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=demucs_model.samplerate)
                wav = resampler(wav)
                sr = demucs_model.samplerate

            wav = wav.to(device)
            
            self.progress.emit("Separating vocals with Demucs...")
            with torch.no_grad():
                # apply_model processes the raw tensor directly
                sources = apply_model(demucs_model, wav.unsqueeze(0), split=True)[0]
            
            vocals_idx = demucs_model.sources.index("vocals")
            vocals_tensor = sources[vocals_idx].cpu()
            
            vocals_path = "temp_vocals.wav"
            torchaudio.save(vocals_path, vocals_tensor, sr)

            # --- Step 2: Transcription & Alignment with WhisperX ---
            self.progress.emit("Loading WhisperX model...")
            model = whisperx.load_model("large-v2", device, compute_type=compute_type)
            
            self.progress.emit("Transcribing vocals...")
            audio = whisperx.load_audio(vocals_path)
            transcribe_result = model.transcribe(audio, batch_size=8)
            
            self.progress.emit("Aligning audio to timestamps...")
            model_a, metadata = whisperx.load_align_model(language_code="ru", device=device)
            aligned_result = whisperx.align(
                transcribe_result["segments"], 
                model_a, 
                metadata, 
                audio, 
                device, 
                return_char_alignments=False
            )

            # --- Step 3: Mapping to Imported Lyrics ---
            self.progress.emit("Mapping to imported text file...")
            
            with open(self.lyrics_path, 'r', encoding='utf-8') as f:
                user_lyrics = [line.strip() for line in f.readlines() if line.strip()]

            final_segments = []
            ai_segments = aligned_result["segments"]
            
            for i, segment in enumerate(ai_segments):
                if "start" not in segment or "end" not in segment:
                    continue
                    
                start_ms = int(segment["start"] * 1000)
                end_ms = int(segment["end"] * 1000)
                
                lyric_ru = user_lyrics[i] if i < len(user_lyrics) else segment["text"]
                translation_en = "" 
                
                final_segments.append((start_ms, end_ms, lyric_ru, translation_en))

            if os.path.exists(vocals_path):
                os.remove(vocals_path)

            self.finished.emit(final_segments)

        except Exception as e:
            print("\n" + "="*50)
            print("CRITICAL ERROR IN AI PIPELINE:")
            traceback.print_exc()
            print("="*50 + "\n")
            
            self.progress.emit(f"Error: {str(e)}")
            self.finished.emit([])

# ==========================================
# 2. Main Application Window
# ==========================================
class LanguageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KinoLingo - Russian A2 Audio Aligner")
        self.resize(800, 600)

        self.db_path = "learning_data.db"
        self.init_database()

        # Media Player Setup
        self.media_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.media_player.setAudioOutput(self.audio_output)
        
        # Active Review State
        self.current_segment_id = None
        self.current_end_ts = 0
        self.current_audio_path = ""
        self.current_expected_translation = ""

        self.setup_ui()

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS songs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            title TEXT, audio_path TEXT)''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS segments (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            song_id INTEGER, start_ts INTEGER, end_ts INTEGER,
                            lyric_ru TEXT, translation_en TEXT,
                            FOREIGN KEY(song_id) REFERENCES songs(id))''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS scores (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            segment_id INTEGER, ease_factor REAL DEFAULT 2.5,
                            interval INTEGER DEFAULT 0, next_review_date DATE,
                            FOREIGN KEY(segment_id) REFERENCES segments(id))''')
        conn.commit()
        conn.close()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Navigation
        nav_layout = QHBoxLayout()
        self.btn_library = QPushButton("Library")
        self.btn_review = QPushButton("Review Mode")
        nav_layout.addWidget(self.btn_library)
        nav_layout.addWidget(self.btn_review)
        main_layout.addLayout(nav_layout)

        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)

        # Pages
        self.page_library = QWidget()
        self.setup_library_page()
        self.stacked_widget.addWidget(self.page_library)

        self.page_review = QWidget()
        self.setup_review_page()
        self.stacked_widget.addWidget(self.page_review)

        # Connect Navigation
        self.btn_library.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.btn_review.clicked.connect(self.switch_to_review_mode)

    def setup_library_page(self):
        layout = QVBoxLayout(self.page_library)
        
        layout.addWidget(QLabel("Your Imported Songs:"))
        self.song_list = QListWidget()
        layout.addWidget(self.song_list)

        # --- New Structured Import Form ---
        layout.addWidget(QLabel("--- Import New Song ---"))
        
        # Audio Selection
        audio_layout = QHBoxLayout()
        self.btn_select_audio = QPushButton("1. Select Audio")
        self.btn_select_audio.clicked.connect(self.select_audio)
        self.lbl_audio_path = QLabel("No audio selected")
        audio_layout.addWidget(self.btn_select_audio)
        audio_layout.addWidget(self.lbl_audio_path)
        layout.addLayout(audio_layout)

        # Lyrics Selection
        lyrics_layout = QHBoxLayout()
        self.btn_select_lyrics = QPushButton("2. Select Lyrics (.txt)")
        self.btn_select_lyrics.clicked.connect(self.select_lyrics)
        self.lbl_lyrics_path = QLabel("No lyrics selected")
        lyrics_layout.addWidget(self.btn_select_lyrics)
        lyrics_layout.addWidget(self.lbl_lyrics_path)
        layout.addLayout(lyrics_layout)

        # Process Button
        self.btn_process = QPushButton("3. Process & Align")
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.start_processing)
        layout.addWidget(self.btn_process)

        self.lbl_import_status = QLabel("")
        layout.addWidget(self.lbl_import_status)

        self.audio_path_pending = None
        self.lyrics_path_pending = None

        self.load_song_list()

    def setup_review_page(self):
        layout = QVBoxLayout(self.page_review)
        
        self.lbl_instruction = QLabel("Load a segment to begin reviewing.")
        layout.addWidget(self.lbl_instruction)

        self.btn_play_segment = QPushButton("▶ Play Audio Segment")
        self.btn_play_segment.clicked.connect(self.play_current_segment)
        layout.addWidget(self.btn_play_segment)

        self.txt_translation = QLineEdit()
        self.txt_translation.setPlaceholderText("Type English translation here...")
        layout.addWidget(self.txt_translation)

        self.btn_submit = QPushButton("Submit Translation")
        self.btn_submit.clicked.connect(self.check_translation)
        layout.addWidget(self.btn_submit)
        
        self.media_player.positionChanged.connect(self.check_audio_position)

    def select_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Song Audio", "", "Audio Files (*.mp3 *.wav *.flac)")
        if path:
            self.audio_path_pending = path
            self.lbl_audio_path.setText(os.path.basename(path))
            self.check_ready_to_process()
    def select_lyrics(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Russian Lyrics", "", "Text Files (*.txt)")
        if path:
            self.lyrics_path_pending = path
            self.lbl_lyrics_path.setText(os.path.basename(path))
            self.check_ready_to_process()

    def check_ready_to_process(self):
        if self.audio_path_pending and self.lyrics_path_pending:
            self.btn_process.setEnabled(True)

    def start_processing(self):
        # Disable all inputs while the AI is working
        self.btn_process.setEnabled(False)
        self.btn_select_audio.setEnabled(False)
        self.btn_select_lyrics.setEnabled(False)
        
        self.title_pending = os.path.basename(self.audio_path_pending)
        
        # Start background thread with actual AI worker
        self.worker = AIAlignmentWorker(self.audio_path_pending, self.lyrics_path_pending)
        self.worker.progress.connect(lambda msg: self.lbl_import_status.setText(msg))
        self.worker.finished.connect(self.save_alignment_to_db)
        self.worker.start()

    # ==========================================
    # 3. Threaded AI Import Logic
    # ==========================================
    def import_song_workflow(self):
        audio_path, _ = QFileDialog.getOpenFileName(self, "Select Song Audio", "", "Audio Files (*.mp3 *.wav)")
        if not audio_path: return
            
        lyrics_path, _ = QFileDialog.getOpenFileName(self, "Select Russian Lyrics", "", "Text Files (*.txt)")
        if not lyrics_path: return

        self.btn_import.setEnabled(False)
        self.title_pending = os.path.basename(audio_path)
        self.audio_path_pending = audio_path
        
        # Start background thread
        self.worker = AIAlignmentWorker(audio_path, lyrics_path)
        self.worker.progress.connect(lambda msg: self.lbl_import_status.setText(msg))
        self.worker.finished.connect(self.save_alignment_to_db)
        self.worker.start()

    @pyqtSlot(list)
    def save_alignment_to_db(self, segments_data):
        if not segments_data:
            self.lbl_import_status.setText("Import failed. Check console for errors.")
            self.btn_process.setEnabled(True)
            return

        self.lbl_import_status.setText("Import Complete!")
        
        # FIX: Reset the new UI buttons instead of the old btn_import
        self.lbl_audio_path.setText("No audio selected")
        self.lbl_lyrics_path.setText("No lyrics selected")
        self.audio_path_pending = None
        self.lyrics_path_pending = None
        self.btn_process.setEnabled(False) 
        self.btn_select_audio.setEnabled(True)
        self.btn_select_lyrics.setEnabled(True)
        
        # Database Insertion Logic
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert Song
        cursor.execute("INSERT INTO songs (title, audio_path) VALUES (?, ?)", 
                       (self.title_pending, self.audio_path_pending))
        song_id = cursor.lastrowid
        
        # Insert Segments and initial SRS scores
        today = datetime.date.today().isoformat()
        for start_ts, end_ts, lyric, trans in segments_data:
            cursor.execute('''INSERT INTO segments (song_id, start_ts, end_ts, lyric_ru, translation_en) 
                              VALUES (?, ?, ?, ?, ?)''', (song_id, start_ts, end_ts, lyric, trans))
            segment_id = cursor.lastrowid
            cursor.execute('''INSERT INTO scores (segment_id, next_review_date) 
                              VALUES (?, ?)''', (segment_id, today))
        
        conn.commit()
        conn.close()
        self.load_song_list()

    def load_song_list(self):
        self.song_list.clear()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT title FROM songs")
        for row in cursor.fetchall():
            self.song_list.addItem(row[0])
        conn.close()

    # ==========================================
    # 4. Review, Playback, and SRS Logic
    # ==========================================
    def switch_to_review_mode(self):
        self.stacked_widget.setCurrentIndex(1)
        self.load_next_review()

    def load_next_review(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        today = datetime.date.today().isoformat()
        
        # Fetch the oldest due segment
        cursor.execute('''
            SELECT s.id, s.start_ts, s.end_ts, s.translation_en, so.audio_path, sc.ease_factor, sc.interval
            FROM segments s
            JOIN scores sc ON s.id = sc.segment_id
            JOIN songs so ON s.song_id = so.id
            WHERE sc.next_review_date <= ?
            ORDER BY sc.next_review_date ASC
            LIMIT 1
        ''', (today,))
        row = cursor.fetchone()
        conn.close()

        if row:
            (self.current_segment_id, self.current_start_ts, self.current_end_ts, 
             self.current_expected_translation, self.current_audio_path, 
             self.current_ease, self.current_interval) = row
            
            self.lbl_instruction.setText("Segment loaded. Press play and translate.")
            self.btn_play_segment.setEnabled(True)
            self.btn_submit.setEnabled(True)
        else:
            self.lbl_instruction.setText("No reviews due today! You are all caught up.")
            self.current_segment_id = None
            self.btn_play_segment.setEnabled(False)
            self.btn_submit.setEnabled(False)

    def play_current_segment(self):
        if not self.current_segment_id: return
        
        # Note: Set your actual audio file path here for testing if the DB path is invalid
        if os.path.exists(self.current_audio_path):
            self.media_player.setSource(QUrl.fromLocalFile(self.current_audio_path))
            self.media_player.setPosition(self.current_start_ts)
            self.media_player.play()
        else:
            QMessageBox.warning(self, "Warning", "Audio file not found at path stored in DB.")

    #@pyqtSlot(int) # Remove the decorator since C++ type error
    def check_audio_position(self, position):
        if self.current_end_ts > 0 and position >= self.current_end_ts:
            self.media_player.stop()

    def check_translation(self):
        if not self.current_segment_id: return
        
        user_input = self.txt_translation.text().strip()
        
        # Fuzzy string matching (0.0 to 1.0)
        similarity = SequenceMatcher(None, user_input.lower(), self.current_expected_translation.lower()).ratio()
        
        # Grading Logic
        if similarity > 0.85:
            grade_msg = "Excellent!"
            new_interval = max(1, int(self.current_interval * self.current_ease))
            new_ease = self.current_ease + 0.1
        elif similarity > 0.65:
            grade_msg = "Close enough."
            new_interval = max(1, int(self.current_interval * 1.2))
            new_ease = self.current_ease
        else:
            grade_msg = "Incorrect."
            new_interval = 0  # Reset interval
            new_ease = max(1.3, self.current_ease - 0.2)
            
        QMessageBox.information(self, "Result", 
                                f"{grade_msg}\n\n"
                                f"Expected: {self.current_expected_translation}\n"
                                f"Similarity Score: {similarity:.2f}")
        
        # Update SRS in Database
        next_date = (datetime.date.today() + datetime.timedelta(days=new_interval)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''UPDATE scores 
                          SET ease_factor = ?, interval = ?, next_review_date = ? 
                          WHERE segment_id = ?''', 
                       (new_ease, new_interval, next_date, self.current_segment_id))
        conn.commit()
        conn.close()

        self.txt_translation.clear()
        self.load_next_review() # Load the next item immediately

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LanguageApp()
    window.show()
    sys.exit(app.exec())
