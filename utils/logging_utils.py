import csv
import os

def log_decision_csv(csv_path: str, timestamp: str, agent_name: str, input_signal: str, reward: float, confidence: float, final_score: float, decision_trace: str):
    """Log decision details to CSV."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'agent_name', 'input_signal', 'reward', 'confidence', 'final_score', 'decision_trace'])
        writer.writerow([timestamp, agent_name, input_signal, reward, confidence, final_score, decision_trace])