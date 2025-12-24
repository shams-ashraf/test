import streamlit as st

def load_custom_css():
        
    # Configuration
    st.set_page_config(
        page_title="MBE Document Assistant - RAG Chatbot",
        page_icon="🎓",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border-left: 4px solid;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left-color: #2196F3;
        }
        .assistant-message {
            background-color: #f5f5f5;
            border-left-color: #4CAF50;
        }
        .source-box {
            background-color: #fff3e0;
            padding: 0.8rem;
            border-radius: 0.3rem;
            margin-top: 0.5rem;
            font-size: 0.85rem;
        }
        .chat-tab {
            display: flex;
            align-items: center;
            padding: 8px 12px;
            margin: 2px;
            border-radius: 5px;
            background-color: #f0f0f0;
            cursor: pointer;
        }
        .chat-tab:hover {
            background-color: #e0e0e0;
        }
        .chat-tab.active {
            background-color: #4CAF50;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
