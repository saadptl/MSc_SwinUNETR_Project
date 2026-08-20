FIXED STREAMLIT DASHBOARD - V2

Main fix:
StreamlitAPIException caused by changing st.session_state.navigation after the
sidebar radio widget with key="navigation" had already been instantiated.

Navigation now works with a two-step mechanism:
1. Buttons write to pending_navigation.
2. main.py consumes pending_navigation BEFORE creating the sidebar radio.

Also fixed the MRI Analysis direct navigation to Results to use the same safe
mechanism.

Replace the corresponding app files in the project, then run:

.\venv\Scripts\python.exe -m streamlit run app\main.py

Do not replace the trained model checkpoint.
