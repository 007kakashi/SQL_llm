import streamlit as st
from main import get_few_shot_ans

st.header('Chat With Database')

input = st.text_input('Ask Your Queries')

answer = st.button('Answer')

if answer:
    ans = get_few_shot_ans()
    get_ans = ans.run(input)
    st.write(get_ans)
