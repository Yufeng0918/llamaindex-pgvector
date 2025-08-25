import uuid
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class DocumentChunk(Base):
    __tablename__ = "my_documents_chunks" # Updated table name to match user's request
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4) # Changed to UUID
    embedding = Column(Vector(1536))
    document_id = Column(String) # Removed index=True as it's not in the provided SQL
    metadata_ = Column('metadata', JSONB) # Renamed to avoid Python keyword conflict, maps to 'metadata' column
    text = Column(Text)
    file_name = Column(String)
    chunk_sequence = Column(Integer) # Renamed to match user's SQL
    active_ind = Column(Boolean, default=True)
    create_timestamp = Column(TIMESTAMP(timezone=True), default=func.now())
    # Removed 'author' as it was not in the provided SQL

    # Added a unique constraint to ensure document_id + chunk_sequence is unique
    # __table_args__ = (
    #     UniqueConstraint('document_id', 'chunk_sequence', name='_document_chunk_uc'),
    # )

    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id='{self.document_id}', chunk_seq={self.chunk_seq})>"
