import os
import streamlit as st
from transformers import GPTJForCausalLM, GPT2Tokenizer
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

# Load the GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTJForCausalLM.from_pretrained(model_name)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Sample document collection (replace with actual document retrieval logic)
documents = [
    "This is a sample document about data science.",
    "Another document talks about machine learning and AI."
]

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = [chunk for doc in documents for chunk in text_splitter.split(doc)]

# Create a FAISS vector store from chunks
vector_store = FAISS.from_texts(chunks)

# Set up the retrieval-based QA system
def create_qa_system():
    try:
        retriever = vector_store.as_retriever()
        qa_system = RetrievalQA(
            retriever=retriever,
            llm=lambda prompt: generate_response(prompt),
        )
        return qa_system
    except Exception as e:
        st.error(f"Error creating QA system: {e}")
        return None

# Define a function to answer user queries
def answer_query(query, qa_system):
    if not qa_system:
        return "QA system is not available."
    try:
        response = qa_system.run(query)
        return response['answer']
    except Exception as e:
        st.error(f"Error answering query: {e}")
        return "An error occurred while processing your query."

# Define a function to handle user information collection
def collect_user_info():
    with st.form(key='user_info_form'):
        st.subheader("Please provide your contact information")
        name = st.text_input("Name")
        phone_number = st.text_input("Phone Number")
        email = st.text_input("Email")
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            st.success("Thank you! We will call you shortly.")
            return name, phone_number, email
        return None

def main():
    st.title("Chatbot")

    # Create QA system
    qa_system = create_qa_system()

    user_query = st.text_input("Ask me anything:")

    if user_query:
        if "call me" in user_query.lower():
            user_info = collect_user_info()
            if user_info:
                name, phone_number, email = user_info
                st.write(f"Name: {name}")
                st.write(f"Phone Number: {phone_number}")
                st.write(f"Email: {email}")
        else:
            response = answer_query(user_query, qa_system)
            st.write(response)

if __name__ == "__main__":
    main()
