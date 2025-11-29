# corefrencing_transformers.py
import os
import json
import torch
import transformers
from typing import List, Dict, Any
import time
from pathlib import Path

class TransformerCoreferenceProcessor:
    """Coreference resolution using Hugging Face transformers"""
    
    def __init__(self, output_dir: str = "processed_docs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            # Method 1: Try coreference resolution model
            print("🔄 Loading coreference resolution model...")
            try:
                # Using a model specifically trained for coreference resolution
                self.coref_pipeline = pipeline(
                    "token-classification",
                    model="coref-hoi/model",
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.method = "coref_model"
                
            except Exception as e:
                print(f"⚠ Coreference model not available, using NER as fallback: {e}")
                # Method 2: Fallback to NER for entity resolution
                self.coref_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.method = "ner_fallback"
            
            self.coref_available = True
            print(f"✅ Transformer pipeline initialized using {self.method}")
            
        except Exception as e:
            print(f"❌ Transformer pipeline failed: {e}")
            self.coref_available = False
    
    def load_document_text(self, file_path: str) -> str:
        """Load document text from various file types"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = file_path.split('.')[-1].lower()
        
        if file_ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_ext == 'pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                print("⚠ Install PyPDF2 for PDF support: pip install PyPDF2")
                return ""
        
        elif file_ext == 'docx':
            try:
                from docx import Document
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                print("⚠ Install python-docx for DOCX support: pip install python-docx")
                return ""
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def resolve_coreferences_transformers(self, text: str) -> Dict[str, Any]:
        """Apply coreference resolution using transformers"""
        if not self.coref_available:
            return {
                "resolved_text": text,
                "coref_applied": False,
                "error": "Transformers pipeline not available"
            }
        
        try:
            start_time = time.time()
            
            if self.method == "coref_model":
                # Use coreference resolution model
                result = self._resolve_with_coref_model(text)
            else:
                # Use NER-based resolution
                result = self._resolve_with_ner(text)
            
            processing_time = time.time() - start_time
            
            return {
                "resolved_text": result["resolved_text"],
                "coref_applied": True,
                "processing_time": processing_time,
                "original_length": len(text),
                "resolved_length": len(result["resolved_text"]),
                "method_used": self.method,
                "entities_found": result.get("entities_count", 0)
            }
            
        except Exception as e:
            return {
                "resolved_text": text,
                "coref_applied": False,
                "error": str(e)
            }
    
    def _resolve_with_coref_model(self, text: str) -> Dict[str, Any]:
        """Resolve coreferences using coreference model"""
        # For coreference models, we process the text directly
        # Note: Actual coreference models would have more complex processing
        entities = self.coref_pipeline(text)
        
        # Simple replacement based on entity continuity
        resolved_text = self._apply_entity_continuity(text, entities)
        
        return {
            "resolved_text": resolved_text,
            "entities_count": len(entities)
        }
    
    def _resolve_with_ner(self, text: str) -> Dict[str, Any]:
        """Resolve coreferences using NER as fallback"""
        # Split text into sentences for better processing
        sentences = self._split_into_sentences(text)
        entities = self.coref_pipeline(text)
        
        # Apply entity-based coreference resolution
        resolved_sentences = self._apply_ner_coreference(sentences, entities)
        resolved_text = " ".join(resolved_sentences)
        
        return {
            "resolved_text": resolved_text,
            "entities_count": len(entities)
        }
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _apply_ner_coreference(self, sentences: List[str], entities: List) -> List[str]:
        """Apply simple coreference resolution using NER entities"""
        resolved_sentences = []
        last_person = None
        last_organization = None
        
        for sentence in sentences:
            words = sentence.split()
            resolved_words = []
            
            for word in words:
                # Clean word for comparison
                clean_word = word.strip('.,!?;:"()')
                
                # Check if this word matches any entity
                for entity in entities:
                    entity_word = entity['word'].strip()
                    if clean_word.lower() == entity_word.lower():
                        if entity['entity_group'] == 'PER':
                            last_person = entity_word
                        elif entity['entity_group'] == 'ORG':
                            last_organization = entity_word
                
                # Replace pronouns with last known entities
                if clean_word.lower() in ['he', 'him', 'his'] and last_person:
                    resolved_words.append(last_person)
                elif clean_word.lower() in ['she', 'her'] and last_person:
                    resolved_words.append(last_person)
                elif clean_word.lower() in ['it'] and last_organization:
                    resolved_words.append(last_organization)
                elif clean_word.lower() in ['they', 'them', 'their'] and last_organization:
                    resolved_words.append(last_organization)
                else:
                    resolved_words.append(word)
            
            resolved_sentence = " ".join(resolved_words)
            resolved_sentences.append(resolved_sentence)
        
        return resolved_sentences
    
    def _apply_entity_continuity(self, text: str, entities: List) -> str:
        """Apply entity continuity for coreference resolution"""
        words = text.split()
        resolved_words = []
        
        # Track last entities
        last_person = None
        last_organization = None
        last_location = None
        
        for word in words:
            clean_word = word.strip('.,!?;:"()')
            replaced = False
            
            # Check if current word is an entity
            for entity in entities:
                if clean_word.lower() in entity['word'].lower():
                    entity_type = entity['entity_group']
                    if entity_type == 'PER':
                        last_person = entity['word']
                    elif entity_type == 'ORG':
                        last_organization = entity['word']
                    elif entity_type == 'LOC':
                        last_location = entity['word']
                    break
            
            # Replace pronouns with last known entities
            if not replaced:
                if clean_word.lower() in ['he', 'him', 'his'] and last_person:
                    resolved_words.append(last_person)
                    replaced = True
                elif clean_word.lower() in ['she', 'her'] and last_person:
                    resolved_words.append(last_person)
                    replaced = True
                elif clean_word.lower() in ['it'] and last_organization:
                    resolved_words.append(last_organization)
                    replaced = True
                elif clean_word.lower() in ['they', 'them', 'their']:
                    if last_organization:
                        resolved_words.append(last_organization)
                        replaced = True
                    elif last_person:
                        resolved_words.append(last_person)
                        replaced = True
            
            if not replaced:
                resolved_words.append(word)
        
        return " ".join(resolved_words)
    
    def advanced_coreference_resolution(self, text: str) -> Dict[str, Any]:
        """More advanced coreference resolution using multiple strategies"""
        if not self.coref_available:
            return self.resolve_coreferences_transformers(text)
        
        try:
            start_time = time.time()
            
            # Strategy 1: Direct NER-based resolution
            entities = self.coref_pipeline(text)
            
            # Strategy 2: Sentence-by-sentence processing
            sentences = self._split_into_sentences(text)
            
            # Track entities across sentences
            entity_tracker = {
                'PERSON': None,
                'ORGANIZATION': None,
                'LOCATION': None
            }
            
            resolved_sentences = []
            
            for sentence in sentences:
                resolved_sentence = self._process_sentence_with_entities(
                    sentence, entities, entity_tracker
                )
                resolved_sentences.append(resolved_sentence)
            
            resolved_text = " ".join(resolved_sentences)
            processing_time = time.time() - start_time
            
            return {
                "resolved_text": resolved_text,
                "coref_applied": True,
                "processing_time": processing_time,
                "original_length": len(text),
                "resolved_length": len(resolved_text),
                "method_used": f"advanced_{self.method}",
                "entities_found": len(entities)
            }
            
        except Exception as e:
            return {
                "resolved_text": text,
                "coref_applied": False,
                "error": str(e)
            }
    
    def _process_sentence_with_entities(self, sentence: str, entities: List, entity_tracker: Dict) -> str:
        """Process a single sentence with entity tracking"""
        words = sentence.split()
        resolved_words = []
        
        for word in words:
            clean_word = word.strip('.,!?;:"()')
            replaced = False
            
            # Update entity tracker
            for entity in entities:
                if clean_word.lower() in entity['word'].lower():
                    entity_type = entity['entity_group']
                    if entity_type == 'PER':
                        entity_tracker['PERSON'] = entity['word']
                    elif entity_type == 'ORG':
                        entity_tracker['ORGANIZATION'] = entity['word']
                    elif entity_type == 'LOC':
                        entity_tracker['LOCATION'] = entity['word']
                    break
            
            # Apply coreference replacements
            if clean_word.lower() in ['he', 'him', 'his'] and entity_tracker['PERSON']:
                resolved_words.append(entity_tracker['PERSON'])
                replaced = True
            elif clean_word.lower() in ['she', 'her'] and entity_tracker['PERSON']:
                resolved_words.append(entity_tracker['PERSON'])
                replaced = True
            elif clean_word.lower() in ['it'] and entity_tracker['ORGANIZATION']:
                resolved_words.append(entity_tracker['ORGANIZATION'])
                replaced = True
            elif clean_word.lower() in ['they', 'them', 'their']:
                if entity_tracker['ORGANIZATION']:
                    resolved_words.append(entity_tracker['ORGANIZATION'])
                    replaced = True
                elif entity_tracker['PERSON']:
                    resolved_words.append(entity_tracker['PERSON'])
                    replaced = True
            
            if not replaced:
                resolved_words.append(word)
        
        return " ".join(resolved_words)
    
    def process_and_save_document(self, file_path: str, enable_coref: bool = True, advanced: bool = False) -> Dict[str, str]:
        """Process single document and save"""
        print(f"📄 Processing: {file_path}")
        
        # Load document
        try:
            full_text = self.load_document_text(file_path)
            if not full_text.strip():
                raise ValueError("Document is empty or could not be read")
        except Exception as e:
            print(f"❌ Error loading document: {e}")
            return {}
        
        # Apply coreference resolution
        if enable_coref and self.coref_available:
            print("   Applying transformer coreference resolution...")
            
            if advanced:
                result = self.advanced_coreference_resolution(full_text)
            else:
                result = self.resolve_coreferences_transformers(full_text)
            
            if result["coref_applied"]:
                print(f"   ✅ Coreference completed in {result['processing_time']:.2f}s")
                print(f"   📊 Text length: {result['original_length']} → {result['resolved_length']} characters")
                print(f"   🔧 Method: {result.get('method_used', 'unknown')}")
                print(f"   🏷️ Entities found: {result.get('entities_found', 0)}")
            else:
                print(f"   ⚠ Coreference failed: {result.get('error', 'Unknown error')}")
        else:
            result = {
                "resolved_text": full_text,
                "coref_applied": False,
                "processing_time": 0
            }
            print("   ⏭ Skipping coreference resolution")
        
        # Prepare output data
        output_data = {
            "original_file": file_path,
            "processed_timestamp": time.time(),
            "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "coref_applied": result["coref_applied"],
            "processing_time": result["processing_time"],
            "method_used": result.get("method_used", "none"),
            "entities_found": result.get("entities_found", 0),
            "text_metrics": {
                "original_length": len(full_text),
                "resolved_length": len(result["resolved_text"]),
                "file_type": file_path.split('.')[-1].lower(),
            }
        }
        
        # Generate filenames
        base_name = self._generate_filename(file_path, result["coref_applied"], advanced)
        text_file_path = self.output_dir / f"{base_name}.txt"
        json_file_path = self.output_dir / f"{base_name}.json"
        
        # Save files
        self._save_text_file(text_file_path, result["resolved_text"])
        self._save_json_file(json_file_path, output_data)
        
        print(f"   💾 Saved: {text_file_path.name}")
        
        return {
            "text_file": str(text_file_path),
            "json_file": str(json_file_path),
            "base_name": base_name
        }
    
    def _generate_filename(self, original_path: str, coref_applied: bool, advanced: bool = False) -> str:
        """Generate filename with timestamp"""
        original_name = Path(original_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if coref_applied:
            if advanced:
                suffix = "_advanced_coref"
            else:
                suffix = "_coref"
        else:
            suffix = "_original"
        
        return f"{original_name}{suffix}_{timestamp}"
    
    def _save_text_file(self, file_path: Path, text: str):
        """Save text content"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def _save_json_file(self, file_path: Path, data: Dict):
        """Save metadata as JSON"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def batch_process(self, file_paths: List[str], enable_coref: bool = True, advanced: bool = False) -> List[Dict[str, str]]:
        """Process multiple documents"""
        processed_files = []
        
        for file_path in file_paths:
            try:
                result = self.process_and_save_document(file_path, enable_coref, advanced)
                if result:  # Only append if successful
                    processed_files.append(result)
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
        
        print(f"\n🎉 Processing completed! {len(processed_files)} documents processed.")
        return processed_files

# Test function
def test_transformer_coreference():
    """Test transformer coreference resolution with sample text"""
    processor = TransformerCoreferenceProcessor()
    
    sample_text = """
    John Smith joined Microsoft in 2020. He was hired as a senior developer. 
    The CEO met with him last week. She approved his new project proposal. 
    The team is excited about it and they believe it will be successful.
    Microsoft is headquartered in Redmond. The company announced new products.
    """
    
    print("Testing transformer coreference resolution...")
    result = processor.resolve_coreferences_transformers(sample_text)
    
    if result["coref_applied"]:
        print("✅ Transformer coreference resolution worked!")
        print("Original:", sample_text)
        print("Resolved:", result["resolved_text"])
        print(f"Method: {result.get('method_used', 'unknown')}")
        print(f"Entities found: {result.get('entities_found', 0)}")
    else:
        print("❌ Transformer coreference resolution failed")
        print(f"Error: {result.get('error', 'Unknown error')}")

# Main usage
def main():
    # Initialize processor
    processor = TransformerCoreferenceProcessor("transformer_processed_docs")
    
    # Test with sample text first
    test_transformer_coreference()
    
    print("\n" + "="*60)
    
    # Process actual documents
    documents = [
        "sample.txt",  # Create a sample text file first
    ]
    
    # Filter to only existing files
    existing_docs = [doc for doc in documents if os.path.exists(doc)]
    
    if not existing_docs:
        # Create a sample file for testing
        with open("sample.txt", "w") as f:
            f.write("""Dr. Smith presented the research findings to Google. She explained that the results were significant. 
            Her team had worked on this project for two years. They were very proud of their accomplishment.
            The company will publish their work next month. It will be available in the journal.
            
            John Davis started at Apple last month. He is working as a data scientist.
            His manager said he is doing excellent work. She expects great things from him.
            The company is located in Cupertino. It recently launched new products.""")
        existing_docs = ["sample.txt"]
        print("📝 Created sample.txt for testing")
    
    # Process with basic method
    print("\n--- Basic Coreference Resolution ---")
    basic_files = processor.batch_process(existing_docs, enable_coref=True, advanced=False)
    
    # Process with advanced method
    print("\n--- Advanced Coreference Resolution ---")
    advanced_files = processor.batch_process(existing_docs, enable_coref=True, advanced=True)
    
    # Print results
    print("\n📋 Processing Summary:")
    for file_info in basic_files + advanced_files:
        if "advanced" in file_info['base_name']:
            status = "✓ Advanced Coref"
        elif "coref" in file_info['base_name']:
            status = "✓ Basic Coref"
        else:
            status = "⏭ Original"
        print(f"   {status}: {file_info['base_name']}")

if __name__ == "__main__":
    main()