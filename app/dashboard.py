import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import os
import json
import time
from datetime import datetime, timedelta

# Paths
DATA_DIR = os.getenv("DATA_DIR", "data")
DB_PATH = os.path.join(DATA_DIR, "assistant_core.db")
CSV_PATH = os.path.join(DATA_DIR, "decision_log.csv")

st.title("Unified Cognitive Intelligence Dashboard")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 30s)", value=False)
refresh_interval = 30

# Load data with error handling and cache management using file mtime
@st.cache_data
def load_decision_log():
    """Load decision log with file mtime-based cache invalidation."""
    try:
        if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
            # Include file mtime in cache key for automatic invalidation
            mtime = os.path.getmtime(CSV_PATH)
            cache_key = f"decision_log_{mtime}"

            df = pd.read_csv(CSV_PATH)
            # Convert timestamp to datetime if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading decision log: {e}")
        return pd.DataFrame()

@st.cache_data
def load_db_data():
    """Load database data with file mtime-based cache invalidation."""
    try:
        if not os.path.exists(DB_PATH):
            st.warning("Database file not found. Please ensure the application has been run.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # Include database file mtime in cache key for automatic invalidation
        mtime = os.path.getmtime(DB_PATH)
        cache_key = f"db_data_{mtime}"

        conn = sqlite3.connect(DB_PATH)
        messages = pd.read_sql("SELECT * FROM messages ORDER BY created_at DESC LIMIT 100", conn)
        decisions = pd.read_sql("SELECT * FROM decisions ORDER BY created_at DESC LIMIT 100", conn)
        rl_logs = pd.read_sql("SELECT * FROM rl_logs ORDER BY created_at DESC LIMIT 100", conn)
        feedback = pd.read_sql("SELECT * FROM feedback ORDER BY created_at DESC LIMIT 100", conn)
        conn.close()

        # Convert timestamps
        for df in [messages, decisions, rl_logs, feedback]:
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

        return messages, decisions, rl_logs, feedback
    except Exception as e:
        st.error(f"Error loading database data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Load data
decision_df = load_decision_log()
messages_df, decisions_df, rl_logs_df, feedback_df = load_db_data()

# Cache invalidation button
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# System Status
st.header("System Status")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Messages", len(messages_df) if not messages_df.empty else 0)
with col2:
    st.metric("Total Decisions", len(decisions_df) if not decisions_df.empty else 0)
with col3:
    st.metric("Total RL Logs", len(rl_logs_df) if not rl_logs_df.empty else 0)
with col4:
    st.metric("Total Feedback", len(feedback_df) if not feedback_df.empty else 0)

# Flow Diagram (simple text)
st.header("System Architecture")
st.text("""
📨 Message → 📝 Summarize → 🧠 Process Summary → 🎯 Decision Hub → 🤖 Agent Action → 💬 Feedback
""")

# Reward Trends with improved visualization
st.header("Reward Trends and Confidence Evolution")
if not decision_df.empty and len(decision_df) > 1:
    # Ensure we have enough data points
    df_plot = decision_df.dropna(subset=['timestamp', 'final_score', 'confidence']).copy()
    if len(df_plot) > 1:
        df_plot = df_plot.sort_values('timestamp')

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Score and confidence over time
        ax1.plot(df_plot['timestamp'], df_plot['final_score'], 'b-', label='Final Score', linewidth=2)
        ax1.plot(df_plot['timestamp'], df_plot['confidence'], 'r--', label='Confidence', linewidth=2)
        ax1.set_title('Decision Scores Over Time')
        ax1.set_ylabel('Score/Confidence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Rolling averages
        if len(df_plot) >= 10:
            df_plot['score_ma'] = df_plot['final_score'].rolling(window=10).mean()
            df_plot['conf_ma'] = df_plot['confidence'].rolling(window=10).mean()
            ax2.plot(df_plot['timestamp'], df_plot['score_ma'], 'b-', label='Score MA(10)', linewidth=2)
            ax2.plot(df_plot['timestamp'], df_plot['conf_ma'], 'r--', label='Confidence MA(10)', linewidth=2)
            ax2.set_title('Moving Averages (Window=10)')
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel('Moving Average')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Final Score", f"{df_plot['final_score'].mean():.3f}")
        with col2:
            st.metric("Avg Confidence", f"{df_plot['confidence'].mean():.3f}")
        with col3:
            st.metric("Score Std Dev", f"{df_plot['final_score'].std():.3f}")
    else:
        st.info("Need more data points for meaningful visualization.")
else:
    st.warning("No decision data available yet. Run some decision hub requests to see trends.")

# Agent Influence Chart with better visualization
st.header("Agent Influence Analysis")
if not decision_df.empty:
    agent_counts = decision_df['agent_name'].value_counts()

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        wedges, texts, autotexts = ax.pie(agent_counts, labels=agent_counts.index,
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title('Agent Selection Distribution')
        plt.setp(autotexts, size=10, weight="bold")
        st.pyplot(fig)

    with col2:
        # Bar chart for better comparison
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(range(len(agent_counts)), agent_counts.values)
        ax.set_xticks(range(len(agent_counts)))
        ax.set_xticklabels(agent_counts.index, rotation=45, ha='right')
        ax.set_title('Agent Selection Counts')
        ax.set_ylabel('Number of Selections')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')

        st.pyplot(fig)
else:
    st.info("No agent decision data available yet.")

# Feedback Analysis
st.header("Feedback Analysis")
if not feedback_df.empty:
    # Feedback distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Feedback value distribution
    ax1.hist(feedback_df['feedback_value'], bins=20, alpha=0.7, edgecolor='black')
    ax1.set_title('Feedback Value Distribution')
    ax1.set_xlabel('Feedback Value (-1 to 1)')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)

    # Feedback by target type
    target_counts = feedback_df['target_type'].value_counts()
    ax2.bar(target_counts.index, target_counts.values)
    ax2.set_title('Feedback by Target Type')
    ax2.set_xlabel('Target Type')
    ax2.set_ylabel('Count')

    plt.tight_layout()
    st.pyplot(fig)

    # Feedback summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Feedback", f"{feedback_df['feedback_value'].mean():.3f}")
    with col2:
        st.metric("Positive Feedback %",
                 f"{(feedback_df['feedback_value'] > 0).mean() * 100:.1f}%")
    with col3:
        st.metric("Feedback Count", len(feedback_df))
else:
    st.info("No feedback data available yet.")

# Logs Viewer with filtering
st.header("Recent Activity Logs")

# Filters
col1, col2 = st.columns(2)
with col1:
    hours_back = st.selectbox("Show last N hours:", [1, 6, 24, 72, 168], index=2)
with col2:
    log_type = st.selectbox("Log type:", ["All", "Decisions", "Messages", "RL Logs", "Feedback"], index=0)

# Filter data by time
cutoff_time = datetime.now() - timedelta(hours=hours_back)

if not decisions_df.empty:
    recent_decisions = decisions_df[decisions_df['created_at'] > cutoff_time]
    st.subheader(f"Recent Decisions (last {hours_back}h)")
    if not recent_decisions.empty:
        st.dataframe(recent_decisions)
    else:
        st.info(f"No decisions in the last {hours_back} hours.")

if not messages_df.empty:
    recent_messages = messages_df[messages_df['created_at'] > cutoff_time]
    st.subheader(f"Recent Messages (last {hours_back}h)")
    if not recent_messages.empty:
        st.dataframe(recent_messages[['id', 'trace_id', 'source', 'content', 'created_at']])
    else:
        st.info(f"No messages in the last {hours_back} hours.")

if not rl_logs_df.empty:
    recent_rl = rl_logs_df[rl_logs_df['created_at'] > cutoff_time]
    st.subheader(f"Recent RL Logs (last {hours_back}h)")
    if not recent_rl.empty:
        st.dataframe(recent_rl)
    else:
        st.info(f"No RL logs in the last {hours_back} hours.")

if not feedback_df.empty:
    recent_feedback = feedback_df[feedback_df['created_at'] > cutoff_time]
    st.subheader(f"Recent Feedback (last {hours_back}h)")
    if not recent_feedback.empty:
        st.dataframe(recent_feedback)
    else:
        st.info(f"No feedback in the last {hours_back} hours.")

# Auto-refresh functionality
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()