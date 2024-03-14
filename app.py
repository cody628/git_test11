import streamlit as st # stramlit(배포)
import tiktoken # token(토큰 기준으로 TextSplitter)
from loguru import logger # log(기록 남김)

from langchain.chains import ConversationalRetrievalChain # RetrievalChain(참고 문서와 LLM 연결)
from langchain.chat_models import ChatOpenAI # LLM 모델 불러옴

from langchain.document_loaders import PyPDFLoader #pdf loader
from langchain.document_loaders import Docx2txtLoader # word loader
from langchain.document_loaders import UnstructuredPowerPointLoader # ppt loader

from langchain.text_splitter import RecursiveCharacterTextSplitter # TextSplitter(문서 분할)
from langchain.embeddings import HuggingFaceEmbeddings # embedding 모델 불러옴

from langchain.memory import ConversationBufferMemory # 대화를 메모리에 넣음
from langchain.vectorstores import FAISS # vectorstore(벡터저장소)

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
  # 페이지 탭을 설정
  st.set_page_config(
  page_title="DirChat",
  page_icon=":books:")

  # 페이지 제목 설정
  st.title("_Private Data :red[QA Chat]_ :books:")


  # st.session_state.conversation 을 초기화
  if "conversation" not in st.session_state:
        st.session_state.conversation = None

  # st.session_state.chat_history 를 초기화
  if "chat_history" not in st.session_state:
      st.session_state.chat_history = None

  if "processComplete" not in st.session_state:
      st.session_state.processComplete = None

  # 좌측 사이드바 설정
  with st.sidebar:
    uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    process = st.button("Process")

  # 좌측 사이드바에서 process 버튼 처리
  if process:
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    files_text = get_text(uploaded_files) # 텍스트로 변환
    text_chunks = get_text_chunks(files_text) # 청크로 변환
    vetorestore = get_vectorstore(text_chunks) # 벡터 저장소에 저장
  
    st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) # chain

    st.session_state.processComplete = True
  
  # 챗봇의 첫머리
  if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "AI developer job interviewer", 
                                      "content": "안녕하세요! 간단한 자기소개 부탁드립니다."}]

  # 
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])
  
  history = StreamlitChatMessageHistory(key="chat_messages")

  # Chat logic
  if query := st.chat_input("답을 입력해주세요."):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
      st.markdown(query)

    with st.chat_message("assistant"):
      chain = st.session_state.conversation

      with st.spinner("Thinking..."):
        result = chain({"question": query})
        with get_openai_callback() as cb:
          st.session_state.chat_history = result['chat_history']
        response = result['answer']
        source_documents = result['source_documents']

        st.markdown(response)
        with st.expander("참고 문서 확인"):
          st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
          st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
          st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)

    st.session_state.messages.append({"role": "assistant", "content": response})

# 토큰 개수를 세는 함수
def tiktoken_len(text):
  tokenizer = tiktoken.get_encoding("cl100k_base")
  tokens = tokenizer.encode(text)
  return len(tokens)

# 업로드된 파일을 텍스트화
def get_text(docs):
  doc_list = []
  
  for doc in docs:
    file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
    with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
      file.write(doc.getvalue())
      logger.info(f"Uploaded {file_name}")
    if '.pdf' in doc.name:
      loader = PyPDFLoader(file_name)
      documents = loader.load_and_split()
    elif '.docx' in doc.name:
      loader = Docx2txtLoader(file_name)
      documents = loader.load_and_split()
    elif '.pptx' in doc.name:
      loader = UnstructuredPowerPointLoader(file_name)
      documents = loader.load_and_split()

    doc_list.extend(documents)

  return doc_list

def get_text_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=100,
    length_function=tiktoken_len
  )
  chunks = text_splitter.split_documents(text)
  return chunks

# embedding
def get_vectorstore(text_chunks):
  embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
  )  

  vectordb = FAISS.from_documents(text_chunks, embeddings)
  return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
  llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
  conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    chain_type="stuff", 
    retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
    memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
    get_chat_history=lambda h: h,
    return_source_documents=True,
    verbose = True
  )

  return conversation_chain

if __name__ == '__main__':
  main()