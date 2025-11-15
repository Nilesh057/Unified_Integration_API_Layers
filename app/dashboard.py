import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import os
import json

# Paths
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "assistant_core.db")
CSV_PATH = os.path.join(DATA_DIR, "decision_log.csv")

st.title("Unified Cognitive Intelligence Dashboard")

# Load data
@st.cache_data
def load_decision_log():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame()

@st.cache_data
def load_db_data():
    conn = sqlite3.connect(DB_PATH)
    messages = pd.read_sql("SELECT * FROM messages ORDER BY created_at DESC LIMIT 10", conn)
    decisions = pd.read_sql("SELECT * FROM decisions ORDER BY created_at DESC LIMIT 10", conn)
    rl_logs = pd.read_sql("SELECT * FROM rl_logs ORDER BY created_at DESC LIMIT 10", conn)
    conn.close()
    return messages, decisions, rl_logs

decision_df = load_decision_log()
messages_df, decisions_df, rl_logs_df = load_db_data()

# Flow Diagram (simple text)
st.header("Live Flow Diagram")
st.text("""
Message → Summarize → Process Summary → Decision Hub → Agent Action → Feedback
""")

# Reward Trends
st.header("Reward Trends and Confidence Evolution")
if not decision_df.empty:
    fig, ax = plt.subplots()
    ax.plot(decision_df['timestamp'], decision_df['final_score'], label='Final Score')
    ax.plot(decision_df['timestamp'], decision_df['confidence'], label='Confidence')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Value')
    ax.legend()
    st.pyplot(fig)
else:
    st.write("No decision data available.")

# Agent Influence Chart
st.header("Agent Influence Chart")
if not decision_df.empty:
    agent_counts = decision_df['agent_name'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(agent_counts, labels=agent_counts.index, autopct='%1.1f%%')
    st.pyplot(fig)
else:
    st.write("No agent data available.")

# Logs Viewer
st.header("Logs Viewer")
st.subheader("Latest Decisions")
st.dataframe(decisions_df)

st.subheader("Latest RL Logs")
st.dataframe(rl_logs_df)

st.subheader("Latest Messages")
st.dataframe(messages_df)

st.subheader("Decision Log CSV")
st.dataframe(decision_df)