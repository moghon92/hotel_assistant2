from agent import chat_with_Anna
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.messages import HumanMessage, AIMessage

import os
import json





def main():


    if 'system_msg' not in st.session_state:
        with open('sysmsg.txt', 'r') as file:
            system_msg = file.read()
        st.session_state.system_msg = system_msg

    if 'hotelDoc' not in st.session_state:
        with open('hoteldoc.txt', 'r') as file:
            hotelDoc = file.read()
        st.session_state.hotelDoc = hotelDoc

    if 'chat_history' not in st.session_state:
        with open('chathistory.json', 'r') as file:
            raw_history = json.load(file)

        chat_history = []
        for qa in raw_history:
            if qa['inputText'] is not None and qa['outputText'] is not None:
                chat_history.extend(
                    [
                        HumanMessage(content=qa['inputText']),
                        AIMessage(content=qa['outputText']),
                    ]
                )
        st.session_state.chat_history = chat_history[-4:]


    st.set_page_config(page_title="Chat with Anna", page_icon=":icecream:")
    st.title('Chat with your Anna :icecream:')

    with st.form('myform', clear_on_submit=False):
        user_msg = st.text_input('Question:', placeholder="What's for breakfast?")
        submitted = st.form_submit_button('Submit')
        # st.markdown(st.session_state.chat_history)
        if submitted and user_msg != "":
            with st.spinner('Calculating...'):
                #chat_box = st.empty()
                #stream_handler = StreamHandler(chat_box)
                response, updated_chat = chat_with_Anna(
                    user_msg,
                    st.session_state.hotelDoc,
                    st.session_state.chat_history,
                    st.session_state.system_msg
                )
                st.session_state.chat_history = updated_chat
                st.markdown(response)

if __name__ == '__main__':
    main()