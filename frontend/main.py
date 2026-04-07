import streamlit as st

from backend import main

def _format_sources(context_docs):
    return [
        meta.get('source', 'Unknown')
        for docs in context_docs
        if (meta := getattr(docs, 'metadata', None)) is not None
    ]

st.set_page_config(page_title="LangChain Documentation Helper", page_icon=":books:", layout="centered")
st.title("LangChain Documentation Helper")

with st.sidebar:
    st.subheader("Session State")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.pop('messages', None)
        st.success("Chat history cleared!")
        st.rerun()



if "messages" not in st.session_state:
    st.session_state.messages = [{'role': 'assistant', 
                                  'content': "Hello! I'm your LangChain Documentation Assistant. Ask me anything about LangChain, and I'll do my best to help you!",
                                  'sources': []}]
    
for message in st.session_state.messages:
    role = st.chat_message(message['role'])
    with role:
        st.markdown(message['content'])
        if message.get('sources'):
            with st.expander('sources'):
                for source in message['sources']:
                    st.markdown(f'- {source}')
                
user_input = st.chat_input('Ask question')
if user_input:
    st.session_state.messages.append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    
    with st.chat_message('assistant'):
        try:
            with st.spinner('Retrieving documnents and generating answer...'):
                result = main.main(user_input)
                final_answer = result['answer']
                sources = _format_sources(result['context_docs'])
    
                st.session_state.messages.append({'role': 'assistant', 'content': final_answer, 'sources': sources})
                st.markdown(final_answer)
                if sources:
                    with st.expander('sources'):
                        for source in sources:
                            st.markdown(f'- {source}')
                            
        except Exception as e:
            st.error(f"Failed to generate response: {e}")