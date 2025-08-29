import os
import warnings
from dotenv import load_dotenv
from typing import List, Dict

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from docling.document_converter import DocumentConverter

warnings.filterwarnings("ignore")
load_dotenv()


class RAGPipeline:
    def __init__(self, model_name="llama3-8b-8192", temperature=0.6):
        """Simple RAG pipeline with Groq LLM + HuggingFace embeddings"""
        api_key = os.getenv("Groq_key")
        if not api_key:
            raise ValueError("Groq_key not found in environment variables")

        # Core components
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = ChatGroq(model=model_name, api_key=api_key,
                            temperature=temperature)
        self.text_splitter = SemanticChunker(self.embeddings)
        self.doc_converter = DocumentConverter()

        # Vector store (memory of documents)
        self.db: FAISS | None = None

        # Prompt template
        self.prompt_template = """
        Use the following context to answer the question.
        If you don't know, just say you don't know.

        Context:
        {context}

        Question: {question}

        Helpful Answer:
        """

    def load_documents(self, sources: List[str]):
        """Load documents, split into chunks, and save in FAISS DB"""
        all_chunks = []

        for source in sources:
            print(f"📄 Processing: {source}")
            result = self.doc_converter.convert(source)
            text = result.document.export_to_text()
            chunks = self.text_splitter.split_text(text)

            docs = [Document(page_content=chunk, metadata={
                             "source": source}) for chunk in chunks]
            all_chunks.extend(docs)

        if all_chunks:
            if self.db is None:
                self.db = FAISS.from_documents(all_chunks, self.embeddings)
            else:
                self.db.add_documents(all_chunks)

            print(f"✅ Loaded {len(all_chunks)} chunks into vector store.")

    def ask(self, question: str) -> Dict[str, any]:
        """Ask a question and retrieve answer from documents"""
        if self.db is None:
            return {"answer": "⚠️ No documents loaded yet.", "sources": []}

        retriever = self.db.as_retriever(
            search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)

        context = "\n\n".join(doc.page_content for doc in docs)

        # Fill prompt
        prompt = self.prompt_template.format(
            context=context, question=question)

        response = self.llm.invoke(prompt)

        unique_sources = list({doc.metadata["source"] for doc in docs})
        return {"answer": response.content, "sources": unique_sources}
