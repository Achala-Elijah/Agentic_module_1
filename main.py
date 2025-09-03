from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient
import os
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Set up embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Initialize ChromaDB
client = PersistentClient(path="./chromadb")
collection = client.get_or_create_collection(
    name="ml_publications",
    metadata={"hnsw:space": "cosine"}
)


"""Load Publications"""
def load_research_publications(documents_path: str):
    documents = []
    for file in os.listdir(documents_path):
        if file.endswith(".txt"):
            file_path = os.path.join(documents_path, file)
            try:
                loader = TextLoader(file_path)
                loaded_docs = loader.load()
                for doc in loaded_docs:
                    documents.append((file, doc.page_content))
                #print(f"Successfully loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
    #print(f"\nTotal documents loaded: {len(documents)}")
    return documents



"""Chunk Publications"""
def chunk_research_paper(paper_content: str, title: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(paper_content)
    return [{"content": c, "title": title, "chunk_id": f"{title}_{i}"}
            for i, c in enumerate(chunks)]


"""Embed Documents"""
def embed_documents(documents: list[str]):
    return embeddings_model.embed_documents(documents)



"""Insert Publications into ChromaDB"""
def insert_publications(collection, publications):
    next_id = collection.count()
    for title, publication in publications:
        chunked_publication = chunk_research_paper(publication, title)
        chunk_texts = [c["content"] for c in chunked_publication]
        embeddings = embed_documents(chunk_texts)
        ids = [f"document_{next_id + i}" for i in range(len(chunked_publication))]
        collection.add(
            embeddings=embeddings,
            ids=ids,
            documents=chunk_texts,
            metadatas=[{"title": c["title"], "chunk_id": c["chunk_id"]} for c in chunked_publication]
        )
        next_id += len(chunked_publication)


""""Search Research DB"""
def search_research_db(query: str, collection, embeddings, top_k=5):
    query_vector = embeddings_model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "content": doc,
            "title": results["metadatas"][0][i]["title"],
            "similarity": 1 - results["distances"][0][i]
        }
        for i, doc in enumerate(results["documents"][0])
    ]



"""Question Answering with LLM"""
def answer_research_question(query: str, collection, embeddings, llm):
    relevant_chunks = search_research_db(query, collection, embeddings, top_k=3)
    context = "\n\n".join([f"From {c['title']}:\n{c['content']}" for c in relevant_chunks])

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Based on the following research findings, answer the researcher's question:

Research Context:
{context}

Researcher's Question: {question}

Answer: Provide a comprehensive answer based on the research findings above.
"""
    )
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content, relevant_chunks



"""Main"""
def main():
    # Path to your txt files
    documents_path = "./documents"

    # Load and insert publications
    publications = load_research_publications(documents_path)
    insert_publications(collection, publications)

    # Get environment variable
    api_key = os.getenv("API_KEY")
    #llm = ChatGroq(model="llama3-8b-8192", api_key=api_key)
    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)

    answer, sources = answer_research_question(
        "Types of learning in AI",
        collection,
        embeddings_model,
        llm
    )

    print("AI Answer:", answer)
    print("\nSOURCES:")
    for i, source in enumerate(sources, start=1):
        print(f"- SOURCE {i}\n{source['content']}")



if __name__ == "__main__":
    main()