import streamlit as st
from app import ChatBot

# Set the page title and favicon (optional)
st.set_page_config(page_title="RAAR Chatbot", page_icon="🤖")

st.title("RAAR Chat-bot")

# Initialize session_state if not already initialized
if 'user_input' not in st.session_state:
    st.session_state.user_input = ["You are now interacting with the RAAR-bot..."]

if 'bot_response' not in st.session_state:
    st.session_state.bot_response = ["I'm RAAR Chat-bot, How can I help you?"]



# Layout of input/response containers
input_container = st.container()
response_container = st.container()

# Add custom CSS for chat-style layout
custom_css = """
.input-container {
    padding: 10px;
    margin: 10px;
    text-align: right;
}

.response-container {
    background: #0072B5;
    color: white;
    border-radius: 5px;
    padding: 10px;
    margin: 10px;
    max-width: 70%;
    word-wrap: break-word;
    text-align: left;
}
"""

# Apply custom CSS
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)

# User input
def get_input():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

with input_container:
    user_input = get_input()

# Response output
def generate_response(prompt):
    chatbot = ChatBot('dialog_data.csv', 'productdata.csv', use_llm=True)
    response = chatbot.respond_to_prompt(user_input)
    return response

with response_container:
    
    if user_input:
        response = generate_response(user_input)
        st.session_state.user_input.append(user_input)
        st.session_state.bot_response.append(response)

    if st.session_state['bot_response']:
        for i in range(len(st.session_state['bot_response'])):
            if i < len(st.session_state['user_input']):
                st.markdown(f'<div class="input-container">You: {st.session_state["user_input"][i]}</div>', unsafe_allow_html=True)
            if i < len(st.session_state['bot_response']):
                st.markdown(f'<div class="response-container">Bot: {st.session_state["bot_response"][i]}</div>', unsafe_allow_html=True)

    
