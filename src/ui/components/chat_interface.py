import streamlit as st
import logging
from typing import List, Optional

from ...services.chat_service import ChatService
from ...models.conversation import ChatMessage, ChatResponse

logger = logging.getLogger(__name__)


def render_chat_interface(chat_service: ChatService) -> None:
    """Render the main chat interface."""
    
    st.header("💬 Chat with Documents")
    
    # Initialize session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Get current session
    session = chat_service.get_current_session()
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        # Show conversation history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show sources for assistant messages
                if message["role"] == "assistant" and "sources" in message:
                    render_source_citations(message["sources"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from chat service
                    response = chat_service.send_message(prompt)
                    
                    # Display response
                    st.write(response.content)
                    
                    # Display sources
                    if response.sources:
                        render_source_citations(response.sources)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.content,
                        "sources": response.sources,
                        "response_time": response.response_time
                    })
                
                except Exception as e:
                    error_message = f"I apologize, but I encountered an error: {str(e)}"
                    st.error(error_message)
                    
                    # Add error message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
    
    # Chat controls
    render_chat_controls(chat_service)


def render_source_citations(sources: List) -> None:
    """Render source citations for a response."""
    
    if not sources:
        return
    
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**Source {i}: {source.document_name}**")
            st.markdown(f"*Relevance: {source.relevance_score:.3f}*")
            
            # Show page number if available
            if source.page_number:
                st.caption(f"Page {source.page_number}")
            
            # Show content preview
            with st.container():
                st.text_area(
                    f"Content Preview {i}",
                    value=source.content,
                    height=100,
                    disabled=True,
                    label_visibility="collapsed"
                )
            
            if i < len(sources):
                st.divider()


def render_chat_controls(chat_service: ChatService) -> None:
    """Render chat control buttons."""
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            chat_service.clear_session()
            st.rerun()
    
    with col2:
        if st.button("📊 Chat Stats"):
            stats = chat_service.get_chat_statistics()
            st.info(f"""
            **Chat Statistics:**
            - Total Sessions: {stats['total_sessions']}
            - Total Messages: {stats['total_messages']}
            - User Messages: {stats['user_messages']}
            - Assistant Messages: {stats['assistant_messages']}
            """)
    
    with col3:
        # Show response time for last message if available
        if st.session_state.messages and "response_time" in st.session_state.messages[-1]:
            last_response_time = st.session_state.messages[-1]["response_time"]
            st.caption(f"Last response time: {last_response_time:.2f}s")


def render_conversation_history(chat_service: ChatService) -> None:
    """Render conversation history management."""
    
    st.header("📜 Conversation History")
    
    # Get current session messages
    messages = chat_service.get_session_history()
    
    if not messages:
        st.info("No conversation history yet. Start chatting to see messages here.")
        return
    
    # Display message count
    st.metric("Total Messages", len(messages))
    
    # Show messages in expandable sections
    user_count = 0
    assistant_count = 0
    
    for i, message in enumerate(messages):
        if message.role == "user":
            user_count += 1
            with st.expander(f"👤 User Message {user_count}", expanded=False):
                st.write(message.content)
                st.caption(f"Sent: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        elif message.role == "assistant":
            assistant_count += 1
            with st.expander(f"🤖 Assistant Message {assistant_count}", expanded=False):
                st.write(message.content)
                st.caption(f"Sent: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Export conversation button
    if st.button("📤 Export Conversation"):
        conversation_text = ""
        for message in messages:
            role_emoji = "👤" if message.role == "user" else "🤖"
            conversation_text += f"{role_emoji} {message.role.title()}: {message.content}\n\n"
        
        st.download_button(
            label="Download Conversation",
            data=conversation_text,
            file_name=f"conversation_{message.timestamp.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )