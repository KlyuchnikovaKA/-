import streamlit as st

pages = {
    "Анализ и модель": [st.Page("analysis_and_model.py", title="Анализ и модель")],
    "Презентация": [st.Page("presentation.py", title="Презентация")],
}

selected_page = st.navigation(pages, position="sidebar", expanded=True)
selected_page.run()
