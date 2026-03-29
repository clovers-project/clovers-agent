import json
from pathlib import Path
from collections import deque
from datetime import datetime
from sqlmodel import SQLModel, Session, select, desc
from sqlmodel import Field
from sqlmodel import create_engine, col, or_
from sqlalchemy import Column
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON


class Memory(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str = Field(index=True)
    content: str
    keywords: list[str] = Field(default=[], sa_column=Column(MutableList.as_mutable(SQLiteJSON)))


class DataManager:
    def __init__(self, path: str, note_size: int = 10) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.sqlite_db = self.path.joinpath("clovers_ai.db")
        self.engine = create_engine(f"sqlite:///{self.sqlite_db.as_posix()}")
        SQLModel.metadata.create_all(self.engine)
        self.note_size = note_size
        self.note: dict[str, deque[str]] = {}

    def read_note(self, session_id: str):
        if session_id not in self.note:
            self.note[session_id] = deque(maxlen=self.note_size)
            file = self.path / f"NOTE-{session_id}.json"
            if not file.exists():
                return ""
            try:
                note = json.loads(file.read_text(encoding="utf-8"))
            except Exception:
                file.unlink()
                return ""
            self.note[session_id].extend(note)
        return "\n".join(self.note[session_id])

    def write_note(self, session_id: str, content: str):
        if session_id not in self.note:
            self.note[session_id] = deque(maxlen=self.note_size)
        self.note[session_id].append(content)
        file = self.path / f"NOTE-{session_id}.json"
        with file.open("w", encoding="utf-8") as f:
            json.dump(list(self.note[session_id]), f, ensure_ascii=False)

    def query_memory(self, session_id: str, keywords: list[str], limit: int = 10):
        with Session(self.engine) as session:
            statement = select(Memory).where(Memory.session_id == session_id)
            if keywords:
                statement = statement.where(or_(*(col(Memory.keywords).contains(k) for k in keywords)))
            statement = statement.order_by(desc(Memory.timestamp)).limit(limit)
            return (x.content for x in session.exec(statement).all())

    def save_memory(self, session_id: str, content: str, keywords: list[str]):
        with Session(self.engine) as session:
            session.add(Memory(session_id=session_id, content=content, keywords=keywords))
            session.commit()
