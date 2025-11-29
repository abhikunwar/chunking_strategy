import os
import re
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

class BaseNotebookLMChunker:
    """Base chunker with common functionality"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def _extract_metadata_from_filename(self, filename: str) -> Dict[str, Any]:
        """Extract basic metadata from filename"""
        return {
            "source": filename,
            "file_name": os.path.basename(filename),
            "file_type": filename.split('.')[-1].lower()
        }
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file type"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.split('.')[-1].lower()
        
        if file_ext == 'pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == 'docx':
            loader = Docx2txtLoader(file_path)
        elif file_ext == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        return loader.load()

class SemanticNotebookLMChunker(BaseNotebookLMChunker):
    """Chunker focused on semantic boundaries (paragraphs, sections, headings)"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 150):
        super().__init__(chunk_size, chunk_overlap)
        
        self.semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "
            ]
        )
    
    def _detect_semantic_boundaries(self, text: str) -> List[str]:
        """Detect natural semantic boundaries in text"""
        # Split by major sections first
        sections = re.split(r'\n{3,}', text)
        
        semantic_segments = []
        for section in sections:
            if not section.strip():
                continue
                
            # Further split by paragraphs
            paragraphs = re.split(r'\n\s*\n', section)
            for paragraph in paragraphs:
                if paragraph.strip():
                    semantic_segments.append(paragraph.strip())
        
        return semantic_segments
    
    def chunk_document(self, file_path: str) -> List[Document]:
        """Main method to chunk a document"""
        documents = self.load_document(file_path)
        all_chunks = []
        
        for doc in documents:
            metadata = doc.metadata.copy()
            metadata.update(self._extract_metadata_from_filename(file_path))
            
            segments = self._detect_semantic_boundaries(doc.page_content)
            
            for segment in segments:
                if len(segment) <= self.chunk_size:
                    # Use segment as-is if small enough
                    all_chunks.append(Document(
                        page_content=segment,
                        metadata=metadata.copy()
                    ))
                else:
                    # Split larger segments
                    segment_chunks = self.semantic_splitter.split_text(segment)
                    for chunk_text in segment_chunks:
                        all_chunks.append(Document(
                            page_content=chunk_text,
                            metadata=metadata.copy()
                        ))
        
        return self._add_chunk_metadata(all_chunks)
    
    def _add_chunk_metadata(self, chunks: List[Document]) -> List[Document]:
        """Add sequential metadata to chunks"""
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        return chunks

class AdvancedNotebookLMChunker(SemanticNotebookLMChunker):
    """Advanced chunker with heading detection and structure awareness"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 150):
        super().__init__(chunk_size, chunk_overlap)
    
    def _detect_heading_boundaries(self, text: str) -> List[str]:
        """Enhanced boundary detection that respects headings and structure"""
        
        # Patterns for different heading types
        heading_patterns = [
            r'\n#{1,6}\s+[^\n]+',                    # Markdown headers: #, ##, ###
            r'\n\d+\.\d*(\.\d*)*\s+[^\n]+',          # Numbered sections: 1, 1.1, 1.1.1
            r'\n[A-Z][A-Z\s]{10,}[A-Z]\n',           # ALL CAPS headings
            r'\n\*+\s+[^\n]+\s+\*+\n',               # *** Section *** style
            r'\n-{3,}\n',                            # --- section breaks ---
        ]
        
        # Combine all patterns
        combined_pattern = '|'.join(f'({pattern})' for pattern in heading_patterns)
        
        # Split by headings while preserving them
        segments = []
        current_position = 0
        
        for match in re.finditer(combined_pattern, text, re.MULTILINE):
            # Add content before the heading
            if match.start() > current_position:
                segment = text[current_position:match.start()].strip()
                if segment:
                    segments.append(segment)
            
            # Add the heading itself
            segments.append(match.group().strip())
            current_position = match.end()
        
        # Add remaining content
        if current_position < len(text):
            segment = text[current_position:].strip()
            if segment:
                segments.append(segment)
        
        return segments if segments else [text]
    
    def _detect_semantic_boundaries(self, text: str) -> List[str]:
        """Override with advanced heading detection"""
        # First, try heading-based segmentation
        heading_segments = self._detect_heading_boundaries(text)
        
        # If no headings found, fall back to parent's semantic detection
        if len(heading_segments) <= 1:
            return super()._detect_semantic_boundaries(text)
        
        return heading_segments

class HybridNotebookLMChunker(AdvancedNotebookLMChunker):
    """Hybrid chunker that combines multiple strategies"""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 150):
        super().__init__(chunk_size, chunk_overlap)
    
    def chunk_document(self, file_path: str) -> List[Document]:
        """Hybrid approach with multiple chunking strategies"""
        documents = self.load_document(file_path)
        all_chunks = []
        
        for doc in documents:
            metadata = doc.metadata.copy()
            metadata.update(self._extract_metadata_from_filename(file_path))
            
            # Get segments using advanced detection
            segments = self._detect_semantic_boundaries(doc.page_content)
            
            for segment in segments:
                chunk_size = self._determine_optimal_chunk_size(segment)
                
                if len(segment) <= chunk_size:
                    all_chunks.append(Document(
                        page_content=segment,
                        metadata=metadata.copy()
                    ))
                else:
                    # Use dynamic chunk size for this segment
                    dynamic_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
                    )
                    segment_chunks = dynamic_splitter.split_text(segment)
                    for chunk_text in segment_chunks:
                        all_chunks.append(Document(
                            page_content=chunk_text,
                            metadata=metadata.copy()
                        ))
        
        return self._add_chunk_metadata(all_chunks)
    
    def _determine_optimal_chunk_size(self, text: str) -> int:
        """Dynamically adjust chunk size based on content characteristics"""
        base_size = self.chunk_size
        
        # Adjust based on content density
        lines = text.split('\n')
        avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0
        
        if avg_line_length > 100:  # Dense text (academic, technical)
            return min(800, base_size)  # Smaller chunks for dense content
        elif avg_line_length < 50:  # Sparse text (dialogue, notes)
            return min(1500, base_size)  # Larger chunks for sparse content
        else:
            return base_size




def demonstrate_all_chunkers():
    file_path = "sample_document.pdf"
    
    # 1. Basic Semantic Chunker
    print("=== Semantic Chunker ===")
    semantic_chunker = SemanticNotebookLMChunker(chunk_size=1024)
    semantic_chunks = semantic_chunker.chunk_document(file_path)
    print(f"Semantic chunks: {len(semantic_chunks)}")
    
    # 2. Advanced Chunker (with heading detection)
    print("\n=== Advanced Chunker ===")
    advanced_chunker = AdvancedNotebookLMChunker(chunk_size=1024)
    advanced_chunks = advanced_chunker.chunk_document(file_path)
    print(f"Advanced chunks: {len(advanced_chunks)}")
    
    # 3. Hybrid Chunker (adaptive chunk sizes)
    print("\n=== Hybrid Chunker ===")
    hybrid_chunker = HybridNotebookLMChunker(chunk_size=1024)
    hybrid_chunks = hybrid_chunker.chunk_document(file_path)
    print(f"Hybrid chunks: {len(hybrid_chunks)}")
    
    # Compare results
    for i, chunk in enumerate(hybrid_chunks[:3]):  # Show first 3 chunks
        print(f"\nChunk {i+1}:")
        print(f"Content: {chunk.page_content[:100]}...")
        print(f"Metadata: {chunk.metadata}")

# Choose the right chunker for your use case
def get_chunker_for_document_type(doc_type: str):
    """Factory function to get appropriate chunker"""
    chunkers = {
        "academic": SemanticNotebookLMChunker(chunk_size=800),   # Dense content
        "technical": AdvancedNotebookLMChunker(chunk_size=1024), # Structured docs
        "general": HybridNotebookLMChunker(chunk_size=1024),     # Mixed content
        "notes": HybridNotebookLMChunker(chunk_size=1500),       # Sparse content
    }
    return chunkers.get(doc_type, HybridNotebookLMChunker())

# Example usage
chunker = get_chunker_for_document_type("academic")
chunks = chunker.chunk_document(r"F:\ryzen-ai-doc\attention_all_you_need.pdf")

print(f"len of chunk is:{len(chunks)}")
print("first 10 chunks")
for i in range(10):
    print(f"chunk : {i}")
    print(chunks[i])