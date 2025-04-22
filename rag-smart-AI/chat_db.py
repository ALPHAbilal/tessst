# chat_db.py
import sqlite3
import logging
import uuid
import json
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple

# --- Configuration ---
DB_PATH = "chat_database.db" # Use a separate DB file

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Database Handler Class ---
class ChatHistoryDB:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._initialize_db()

    @contextmanager
    def _get_conn(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            conn.row_factory = sqlite3.Row # Return rows as dict-like objects
            yield conn
        except Exception as e:
            logger.error(f"Database connection/operation error: {e}", exc_info=True)
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.commit() # Commit changes if no exceptions occurred
                conn.close()

    def _initialize_db(self):
        """Create tables if they don't exist."""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                # Chats table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    vector_store_id TEXT,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                ''')
                # Messages table
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL, -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
                )
                ''')
                # Chat files table - for file inclusion settings
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    included BOOLEAN NOT NULL DEFAULT 1,
                    FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE,
                    UNIQUE(chat_id, file_id)
                )
                ''')
                # Index for faster message retrieval
                cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id_timestamp
                ON chat_messages (chat_id, timestamp);
                ''')
                # Index for faster chat files retrieval
                cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_chat_files_chat_id
                ON chat_files (chat_id);
                ''')
                logger.info("Chat database schema initialized/verified.")
        except Exception as e:
            logger.error(f"Error initializing chat database schema: {e}", exc_info=True)
            pass # Allow app to continue, but features might fail

    def get_chats(self, limit: int = 50) -> List[Dict]:
        """Retrieves a list of recent chats."""
        chats = []
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, title, vector_store_id, updated_at FROM chats ORDER BY updated_at DESC LIMIT ?", (limit,))
                chats = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error fetching chats: {e}", exc_info=True)
        return chats

    def create_chat(self, vector_store_id: str, title: str) -> Optional[str]:
        """Creates a new chat session and returns its ID."""
        chat_id = str(uuid.uuid4())
        now = datetime.now()
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO chats (id, vector_store_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (chat_id, vector_store_id, title, now, now)
                )
                logger.info(f"Created new chat with ID: {chat_id}")
                return chat_id
        except Exception as e:
            logger.error(f"Error creating chat: {e}", exc_info=True)
            return None

    def add_message(self, chat_id: str, role: str, content: str) -> bool:
        """Adds a message to a specific chat and updates the chat's timestamp."""
        now = datetime.now()
        success = False
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO chat_messages (chat_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                    (chat_id, role, content, now)
                )
                cursor.execute(
                    "UPDATE chats SET updated_at = ? WHERE id = ?",
                    (now, chat_id)
                )
                logger.debug(f"Added '{role}' message to chat {chat_id}")
                success = True
        except Exception as e:
            logger.error(f"Error adding message to chat {chat_id}: {e}", exc_info=True)
        return success

    def get_messages(self, chat_id: str, limit: int = 50) -> List[Dict]:
        """Retrieves messages for a specific chat, ordered by timestamp."""
        messages = []
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT role, content, timestamp FROM chat_messages WHERE chat_id = ? ORDER BY timestamp DESC LIMIT ?", # Get latest first
                    (chat_id, limit)
                )
                # Reverse in Python to get ascending order for display/API context
                messages = [dict(row) for row in reversed(cursor.fetchall())]
        except Exception as e:
            logger.error(f"Error fetching messages for chat {chat_id}: {e}", exc_info=True)
        return messages

    def get_chat_details(self, chat_id: str) -> Optional[Dict]:
        """Retrieves metadata for a specific chat."""
        details = None
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id, title, vector_store_id, created_at, updated_at FROM chats WHERE id = ?", (chat_id,))
                row = cursor.fetchone()
                if row: details = dict(row)
        except Exception as e:
            logger.error(f"Error fetching details for chat {chat_id}: {e}", exc_info=True)
        return details

    def rename_chat(self, chat_id: str, new_title: str) -> bool:
        """Renames a specific chat session."""
        if not new_title or not chat_id: return False
        now = datetime.now(); success = False
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor(); cursor.execute("UPDATE chats SET title = ?, updated_at = ? WHERE id = ?", (new_title.strip(), now, chat_id))
                if cursor.rowcount > 0: success = True; logger.info(f"Renamed chat {chat_id}")
                else: logger.warning(f"Chat {chat_id} not found for renaming.")
        except Exception as e: logger.error(f"Error renaming chat {chat_id}: {e}")
        return success

    def delete_chat(self, chat_id: str) -> bool:
        """Deletes a specific chat session and its messages."""
        if not chat_id: return False
        success = False
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor(); cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
                if cursor.rowcount > 0: success = True; logger.info(f"Deleted chat {chat_id}")
                else: logger.warning(f"Chat {chat_id} not found for deletion.")
        except Exception as e: logger.error(f"Error deleting chat {chat_id}: {e}")
        return success

    def get_chat_files(self, chat_id: str) -> List[Dict]:
        """Retrieves file inclusion settings for a specific chat."""
        files = []
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT file_id, included FROM chat_files WHERE chat_id = ?",
                    (chat_id,)
                )
                files = [dict(row) for row in cursor.fetchall()]
                logger.debug(f"Retrieved {len(files)} file settings for chat {chat_id}")
        except Exception as e:
            logger.error(f"Error fetching file settings for chat {chat_id}: {e}", exc_info=True)
        return files

    def set_chat_files(self, chat_id: str, file_ids: List[str], included: bool) -> bool:
        """Updates file inclusion settings for a specific chat."""
        if not chat_id or not file_ids: return False
        success = False
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                # Use REPLACE to handle the UNIQUE constraint
                for file_id in file_ids:
                    cursor.execute(
                        "REPLACE INTO chat_files (chat_id, file_id, included) VALUES (?, ?, ?)",
                        (chat_id, file_id, included)
                    )
                success = True
                logger.info(f"Updated {len(file_ids)} file settings for chat {chat_id}")
        except Exception as e:
            logger.error(f"Error updating file settings for chat {chat_id}: {e}", exc_info=True)
        return success
