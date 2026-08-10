import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Page Config
st.set_page_config(
    page_title="Simple LangChain Chatbot with Groq",
    page_icon="⭐"
)


# Title
st.title("⭐ Simple LangChain Chat with Groq")
st.markdown("Learn LangChain basics with Groq's ultra-fast inference!")


# Sidebar
with st.sidebar:
    st.header("Settings")

    # API Key
    api_key = st.text_input(
        "Groq API key",
        type="password",
        help="Get a free API key at console.groq.com"
    )

    # Model Selection
    model_name = st.selectbox(
        "Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile"
        ],
        index=0
    )

    # Clear button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []


# Initialize LLM Model
@st.cache_resource
def get_chain(api_key, model_name):

    if not api_key:
        return None

    # Initialize Groq model
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.7,
        streaming=True
    )

    # Create Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant powered by Groq. "
            "Answer questions clearly and concisely."
        ),
        (
            "user",
            "{question}"
        )
    ])

    # Create chain
    chain = prompt | llm | StrOutputParser()

    return chain


# Get chain
chain = get_chain(api_key, model_name)


# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Chat input
question = st.chat_input("Ask me anything")


if question:

    # If API key isn't entered
    if not chain:
        st.warning("☝️ Please enter your Groq API key in the sidebar.")
        st.stop()

    # Add user message to session state
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Display user message
    with st.chat_message("user"):
        st.write(question)

    # Generate response
    with st.chat_message("assistant"):

        message_placeholder = st.empty()
        full_response = ""

        try:

            # Stream response from Groq
            for chunk in chain.stream({
                "question": question
            }):

                full_response += chunk

                message_placeholder.markdown(
                    full_response + "▌"
                )

            # Final response
            message_placeholder.markdown(full_response)

            # Add assistant response to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")


# Examples
st.markdown("---")
st.markdown("### 💡 Try these Examples:")

col1, col2 = st.columns(2)

with col1:
    st.markdown("- What is LangChain?")
    st.markdown("- Explain Groq's LPU technology")

with col2:
    st.markdown("- How do I learn programming?")
    st.markdown("- Write a haiku about AI")


# Footer
st.markdown("---")
st.markdown("Built with LangChain and Groq | Experience the Speed! ✨")