import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

# Load API key securely from Streamlit Secrets
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]



GOOGLE_API_KEY="AIzaSyBXn6ZteZSsARNFPQGRS8a3z1vnPExW-KM"


genai.configure(api_key=GOOGLE_API_KEY)

# 1. Extract PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# 2. Split into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# 3. Create & save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# 4. Handle user input
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=3)

    # Combine content
    context = "\n\n".join([doc.page_content for doc in docs])
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
Answer the following question using only the information provided in the context below.
Do not copy the answer directly from the context — instead, refine and rephrase it in your own words.
If the answer cannot be found in the context, respond with: "Answer not found in the context."

Context:
{{context}}

Question:
{{question}}
    """

    response = model.generate_content(prompt)
    st.write("### 📘 Answer:")
    st.write(response.text)

# 5. Streamlit App
def main():
    st.set_page_config(page_title="Chat with PDF using Gemini")
    st.header("📄 Chat with PDF using Gemini 💁‍♂️")

    user_question = st.text_input("Ask a question about the PDF content:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📁 Upload PDF")
        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Done! You can now ask questions.")

if __name__ == "__main__":
    main()
