from flask import Flask, request, jsonify, render_template
from textblob import TextBlob
import speech_recognition as sr
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import json

app = Flask(__name__)

# Database setup
def init_db():
    conn = sqlite3.connect('helpdesk_sentiment.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS call_sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            call_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            customer_id TEXT,
            transcript TEXT NOT NULL,
            sentiment_score REAL,
            sentiment_label TEXT,
            key_phrases TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            call_duration INTEGER,
            resolution_status TEXT
        )
    ''')
    conn.commit()
    conn.close()

class SentimentAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
    
    def analyze_text(self, text):
        """Analyze sentiment of text using TextBlob"""
        analysis = TextBlob(text)
        
        # Get sentiment polarity (-1 to 1)
        sentiment_score = analysis.sentiment.polarity
        
        # Determine sentiment label
        if sentiment_score > 0.1:
            sentiment_label = 'Positive'
        elif sentiment_score < -0.1:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        
        # Extract key phrases (nouns and adjectives)
        key_phrases = []
        for phrase in analysis.noun_phrases:
            key_phrases.append(phrase)
            
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'key_phrases': key_phrases
        }
    
    def transcribe_audio(self, audio_file):
        """Convert audio to text using speech recognition"""
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                return text
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            return None

class CallAnalytics:
    def __init__(self):
        self.conn = sqlite3.connect('helpdesk_sentiment.db')
        
    def save_analysis(self, analysis_data):
        """Save call analysis to database"""
        c = self.conn.cursor()
        c.execute('''
            INSERT INTO call_sentiments 
            (call_id, agent_id, customer_id, transcript, sentiment_score, 
             sentiment_label, key_phrases, call_duration, resolution_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_data['call_id'],
            analysis_data['agent_id'],
            analysis_data['customer_id'],
            analysis_data['transcript'],
            analysis_data['sentiment_score'],
            analysis_data['sentiment_label'],
            json.dumps(analysis_data['key_phrases']),
            analysis_data['call_duration'],
            analysis_data['resolution_status']
        ))
        self.conn.commit()
    
    def get_agent_performance(self, agent_id, date_from=None, date_to=None):
        """Get sentiment analysis statistics for a specific agent"""
        query = """
            SELECT 
                COUNT(*) as total_calls,
                AVG(sentiment_score) as avg_sentiment,
                SUM(CASE WHEN sentiment_label = 'Positive' THEN 1 ELSE 0 END) as positive_calls,
                SUM(CASE WHEN sentiment_label = 'Negative' THEN 1 ELSE 0 END) as negative_calls,
                AVG(call_duration) as avg_duration
            FROM call_sentiments
            WHERE agent_id = ?
        """
        params = [agent_id]
        
        if date_from and date_to:
            query += " AND timestamp BETWEEN ? AND ?"
            params.extend([date_from, date_to])
            
        c = self.conn.cursor()
        c.execute(query, params)
        return dict(zip(['total_calls', 'avg_sentiment', 'positive_calls', 
                        'negative_calls', 'avg_duration'], c.fetchone()))
    
    def get_trend_analysis(self, days=30):
        """Get sentiment trends over time"""
        query = """
            SELECT 
                DATE(timestamp) as date,
                AVG(sentiment_score) as avg_sentiment,
                COUNT(*) as call_count
            FROM call_sentiments
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, self.conn, params=[days])
        return df.to_dict('records')

# Flask routes
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/analyze_call', methods=['POST'])
def analyze_call():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    analyzer = SentimentAnalyzer()
    
    # Transcribe audio
    transcript = analyzer.transcribe_audio(audio_file)
    if not transcript:
        return jsonify({'error': 'Failed to transcribe audio'}), 400
    
    # Analyze sentiment
    analysis = analyzer.analyze_text(transcript)
    
    # Prepare analysis data
    analysis_data = {
        'call_id': request.form.get('call_id'),
        'agent_id': request.form.get('agent_id'),
        'customer_id': request.form.get('customer_id'),
        'transcript': transcript,
        'sentiment_score': analysis['sentiment_score'],
        'sentiment_label': analysis['sentiment_label'],
        'key_phrases': analysis['key_phrases'],
        'call_duration': int(request.form.get('duration', 0)),
        'resolution_status': request.form.get('status', 'unresolved')
    }
    
    # Save analysis
    analytics = CallAnalytics()
    analytics.save_analysis(analysis_data)
    
    return jsonify(analysis_data)

@app.route('/agent_performance/<agent_id>')
def agent_performance(agent_id):
    date_from = request.args.get('from')
    date_to = request.args.get('to')
    
    analytics = CallAnalytics()
    performance = analytics.get_agent_performance(agent_id, date_from, date_to)
    return jsonify(performance)

@app.route('/sentiment_trends')
def sentiment_trends():
    days = int(request.args.get('days', 30))
    analytics = CallAnalytics()
    trends = analytics.get_trend_analysis(days)
    return jsonify(trends)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)