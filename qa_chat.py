from pypdf import PdfReader
import torch
import PyPDF2
from io import BytesIO
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import gradio as gr
import time

from langchain.memory import ConversationBufferMemory


from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.document_loaders import PyPDFDirectoryLoader

CHUNK_SIZE = 1000
# Using HuggingFaceEmbeddings with the chosen embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",model_kwargs = {"device": "cuda"})

# transformer model configuration
quant_config = BitsAndBytesConfig(
    bnb_4bit_compute_dtype=torch.bfloat16
)


def load_llm():

    model_id = "Deci/DeciLM-6b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                            trust_remote_code=True,
                                            device_map = "auto",
                                            quantization_config=quant_config)
    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    temperature=0,
                    num_beams=5,
                    no_repeat_ngram_size=4,
                    early_stopping=True,
                    max_new_tokens=50,
                )
    
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

def add_text(history, text):
    if not text:
        raise gr.Error('Enter text')
    history = history + [(text, '')]

    return history

def upload_file(file):
    # file_path = [file.name for file in files]
    print(type(file))
    return file

def process_file(files):

   
    
    # loader = PyPDFLoader(file_path= file.name)
    # document = loader.load()

    pdf_text = ""
    for file in files:
        # pdf_stream = BytesIO(file.name.content)
        pdf = PyPDF2.PdfReader(file.name)
        for page in pdf.pages:
            pdf_text += page.extract_text()

    



    

        
    # split into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=200)

    splits = text_splitter.create_documents([pdf_text])

    # create a FAISS vector store db 

    # embedd the chunks and store in the db
    vectorstore_db = FAISS.from_documents(splits, embeddings)

    #create a custom prompt
    custom_prompt_template = """Given the uploaded files, generate a pecise answer to the question asked by the user.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Context= {context}
    History = {history}
    Question= {question}
    Helpful Answer:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["question", "context", "history"])
    

    # set QA chain with memory
    qa_chain_with_memory = RetrievalQA.from_chain_type(llm=load_llm(),
                                                       chain_type='stuff',
                                                       return_source_documents=True,
                                                       retriever=vectorstore_db.as_retriever(),
                                                       chain_type_kwargs={"verbose": True,
                                                                          "prompt": prompt,
                                                                          "memory": ConversationBufferMemory(
                                                                              input_key="question",
                                                                              memory_key="history",
                                                                              return_messages=True) })
    
    # get answers
    return qa_chain_with_memory


def generate_bot_response(history,query, btn):

    if not btn:
        raise gr.Error(message='Upload a PDF')
    
    qa_chain_with_memory = process_file(btn)


    bot_response = qa_chain_with_memory({"query": query})

    # return bot_response["result"]
    for char in bot_response['result']:
           history[-1][-1] += char
           time.sleep(0.05)
           yield history,''

with gr.Blocks() as demo:
    with gr.Row():
            with gr.Row():                 
                chatbot = gr.Chatbot(label="DeciLM-6b-instruct bot", value=[], elem_id='chatbot')
            with gr.Row():
                 file_output = gr.File(label="Your PDFs")
                 with gr.Column():
                    btn = gr.UploadButton("📁 Upload a PDF(s)", file_types=[".pdf"], file_count="multiple")
                    

    with gr.Column():        
        with gr.Column():
            txt = gr.Text(show_label=False, placeholder="Enter question")
 
        with gr.Column():
            submit_btn = gr.Button('Ask')


# Event handler for uploading a PDF
    btn.upload(fn=upload_file, inputs=[btn], outputs=[file_output])


    submit_btn.click(
            fn= add_text,
            inputs=[chatbot, txt],
            outputs=[chatbot],
            queue=False
        ).success(
            fn=generate_bot_response,
            inputs=[chatbot, txt, btn],
            outputs=[chatbot, txt]
        ).success(
            fn=upload_file,
            inputs=[btn],
            outputs=[file_output]
        )

if __name__ == "__main__":
    demo.launch()