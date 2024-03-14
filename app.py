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
    # 페이지 탭의 아이콘과 제목 설정
    st.set_page_config(
    page_title="인터버", # 제목
    page_icon=":books:") # 아이콘

    # 페이지 제목 설정
    st.title("_Private Data :red[QA Chat]_ :books:")

    # st.session_state에 'conversation' 키가 존재하지 않을 경우, 해당 키를 None으로 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None # 사용자와 챗봇 간의 대화 상태를 저장할 수 있는 변수

    # st.session_state에 'chat_history' 키가 존재하지 않을 경우, 해당 키를 None으로 초기화
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None # 대화의 이전 내용을 저장하여, 대화의 맥락을 유지하는 데 사용

    # 사이드바 위젯 설정
    with st.sidebar: # 사이드바 영역에서 위젯을 생성하기 위한 문맥 관리자
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True) # 파일 업로드 위젯 생성
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password") # api키 입력 위젯 생성
        process = st.button("Process") # process 버튼

    # process 버튼 처리   
    if process:
        if not openai_api_key: # api키가 입력되지 않았을 때 처리
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files) # 업로드된 파일에서 text 추출
        text_chunks = get_text_chunks(files_text) # 추출된 text를 chunk(text_splitter) 처리
        vetorestore = get_vectorstore(text_chunks) # 벡터 저장소에 저장

        # 벡터 저장소와 api키를 이용하여 대화 체인 초기화
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True

    # streamlit을 사용하여 챗봇과 대화를 UI에 표시하는 부분
    if 'messages' not in st.session_state: # st.session_state에 message키가 존재하지 않을 경우 이를 초기화하는 과정
        st.session_state['messages'] = [{"role": "Developer job interviewer", 
                                        "content": "안녕하세요! AI 면접 시스템에 오신걸 환영합니다."}]

    # st.session_state.messages에 저장된 모든 메시지를 순회하며, 각 메시지를 사용자 인터페이스에 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): # 메시지의 역할
            st.markdown(message["content"]) # 메시지의 내용
    
    # 대화 기록 관리
    history = StreamlitChatMessageHistory(key="chat_messages")

    # 대화 체인 로직
    # 사용자로 부터 답변 입력 받기
    if answer := st.chat_input("답변을 입력해주세요."):
        st.session_state.messages.append({"role": "interviewee", "content": answer}) # 답변 메시지 st.session_state.messages 리스트에 추가

        # 사용자 답변을 대화창에 표시
        with st.chat_message("interviewee"):
            st.markdown(answer)

        # 챗봇 응답 처리
        with st.chat_message("Developer job interviewer"): # 이전에 설정된 대화 체인(conversation chain)을 로드
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"answer": answer}) # 답변과 로드한 파일을 함께 연결해 응답 생성

                # 응답 및 관련 문서 표시
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                  
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "interviewee", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

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
