from typing import List, Dict, Any
from services.document_processor import DocumentProcessor
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.llm_service import LLMService
from services.intent_analyzer import LocalIntentAnalyzer

from models.schemas import QueryRequest, QueryResponse
from config.settings import settings

class QueryEngine:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.llm_service = LLMService()
        self.intent_analyzer = LocalIntentAnalyzer()

    
    async def process_query(self, request: QueryRequest) -> QueryResponse:
        """Main method to process document and answer questions"""
        import time
        start_time = time.time()
        
        try:
            # Step 1: Process document
            print("Processing document...")
            chunks = await self.doc_processor.process_document(request.documents)
            print(f"Created {len(chunks)} semantic chunks")
            
            # Step 2: Generate embeddings for chunks with metadata context
            print("Generating embeddings with context...")
            texts = [chunk.text for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            embeddings = self.embedding_service.encode_texts(texts, metadatas)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
            
            # Step 3: Clear existing data and store in vector database
            print("Storing in vector database...")
            self.vector_store.clear_index()
            self.vector_store.store_chunks(chunks)
            
            # Step 4: Process each question
            print(f"Processing {len(request.questions)} questions...")
            answers = []
            for i, question in enumerate(request.questions, 1):
                print(f"   Question {i}/{len(request.questions)}: Processing...")
                answer = await self._answer_question(question)
                answers.append(answer)
                print(f"   Question {i} completed")
            
            total_time = time.time() - start_time
            print(f"\nCOMPLETED: {len(chunks)} chunks processed, {len(answers)} answers generated in {total_time:.2f}s")
            

            
            # Complete cleanup - remove all FAISS files
            self.vector_store.complete_cleanup()
            print("FAISS index and metadata files deleted")
            
            return QueryResponse(answers=answers)
            
        except Exception as e:
            # Cleanup on error too
            self.vector_store.complete_cleanup()
            raise Exception(f"Failed to process query: {str(e)}")
    

    


    async def _answer_question(self, question: str) -> str:
        """Answer individual question using enhanced RAG"""
        try:
            # Generate embedding for question
            question_embedding = self.embedding_service.encode_single_text(question)

            # Fast local intent analysis
            query_intent = self.intent_analyzer.analyze_intent(question)
            
            relevant_chunks = self.vector_store.search_similar(
                question_embedding,
                top_k=settings.TOP_K_RETRIEVAL,
                query_text=question,
                query_intent=query_intent
            )

            # Clean output - show retrieved chunks with key info
            intent_type = query_intent.get('intent_type', 'general')
            looking_for = query_intent.get('looking_for', 'information')
            confidence = query_intent.get('confidence', 0.0)
            print(f"      Intent: {intent_type} - {looking_for} (conf: {confidence:.2f})")
            print(f"      Retrieved {len(relevant_chunks)} chunks:")
            for i, chunk in enumerate(relevant_chunks, 1):
                chunk_preview = chunk.chunk.text[:60].replace('\n', ' ') + "..."
                print(f"         {i}. {chunk.score:.3f} | {chunk.chunk.metadata.get('type', 'unknown')} | {chunk_preview}")
            


            # Use retrieved chunks for LLM
            llm_chunks = relevant_chunks
            
            # Generate answer using LLM
            answer = self.llm_service.generate_answer(question, llm_chunks)

            return answer

        except Exception as e:
            return f"Error answering question: {str(e)}"
    