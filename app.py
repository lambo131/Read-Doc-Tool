import streamlit as st
import numpy as np
import function as func

@st.cache_data
def load_resource():
    return func.load_models()

@st.cache_resource
def load_web_data(web_address):
    session.load_web_data(web_address)
    return session

@st.cache_data
def get_answer(question, debug=False):
    if debug:
        str = func.ans_example
    else:
        str = session.get_answer(question)
    return str

@st.cache_data
def get_good_ques(text, debug=False):
    if debug:
        list = func.good_ques_example
    else:
        list = session.get_good_ques(text)
    return list

# initialize variables *******************
llm, openai_embeddings = load_resource()
session = func.Session(llm, openai_embeddings)
debug = False
if debug:
    st.subheader(":red[Debug is On!]",)

col1, col2, col3 = st.columns(3, gap="large")
col3.write("created by Lambo Qin")
st.write(f"test rerun: {np.random.randint(1,100)}")
st.header("Doc Question & Answer Tool :page_with_curl:")

# Load documents *************************************
option = st.selectbox(
    'Choose document source',
    ('web address', 'upload file', 'text input'))
if option == "web address":
    web_address = st.text_input("web address")
    if web_address != "": 
        loaded = False
        # st.write(np.random.randint(1,10))
        session = load_web_data(web_address)
        if session.qa_chain != None and session.load_successful:
            loaded = True
            got_ans = False
if option == "upload file":
    uploaded_file = st.file_uploader("Choose a file")
if option == "text input":
    st.text_input("text input")

st.write("---")
# Question *************************************
question = st.text_input("Enter your question")
button = st.button("Generate") # put button widget before question display
your_question = st.empty()
try:
    if st.session_state.update_ques_input == True:
        your_question.subheader(f"Your question")
        st.markdown(f"**{st.session_state.good_ques_select}**")
    else:
        your_question.subheader(f"your question:")
        st.markdown(f"### {question}")

except: 
    your_question.subheader(f"Your question:")
    st.markdown(f"#### -> _{question}_")
    

# Generate *************************************
left, right = st.columns(2)
left.subheader("Output")
output = left.empty()
words = left.empty()

if button:
    if question != "":
        got_ans = False
        if loaded:
            st.session_state.ans = get_answer(question, debug)
            got_ans = True
        else:
            st.write("thing")
            output.write("no website data!")
    else:
        output.write("Enter a question!")

try:
    if st.session_state.update_ques_input == True:
        got_ans = False
        if loaded:
            st.session_state.ans = get_answer(st.session_state.good_ques_select,debug)
            got_ans = True
        else:
            st.write("thing")
            output.write("no website data!")
except: pass

try:
    output.write(st.session_state.ans)
except:
    output.write("")
try:
    st.session_state.good_ques = get_good_ques(st.session_state.ans, debug)
except:"'ans' not loaded"

# Good questions *************************************
right.subheader("Good questions to ask")
good_ques_list = []
choices = {}

st.markdown(
    """
    <style>
    button {
        height: 1;
        font-size: 14px;
        text-align: left;
        border: none;
        line-height: 2 px;
    }
    </style>
    """,unsafe_allow_html=True,
    )


try:
    # st.session_state.good_ques
    if len(st.session_state.good_ques) != 0:
        for ques in st.session_state.good_ques:
            button = right.button("| "+ques)
            choices[ques] = button
except:"could not create question buttons"

for i, str in enumerate(choices):
    if choices[str]: # if the question suggesting button is clicked
        right.write(f"button {i+1} was clicked")
        right.write(f"ques: {str}")
        st.session_state.good_ques_select = str
        st.session_state.update_ques_input = True
        st.experimental_rerun()

st.write("---")

# debugger options
if debug:
    st.write(session.session_docs)