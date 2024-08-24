import os
import logging
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import spacy
from collections import Counter
import tika
from tika import parser
import base64
import re
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import csv
import io
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import asyncio
from concurrent.futures import ThreadPoolExecutor
import schedule
import time
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
from requests.exceptions import RequestException
import signal
import sys
import threading
import atexit
from memory_profiler import memory_usage
import psutil
from pybreaker import CircuitBreaker, CircuitBreakerError

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Tika
tika.initVM()

# Global variables
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
G = nx.Graph()

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'email_analyzer'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

# Create a connection pool
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=10,
    **DB_CONFIG
)

# Graceful Shutdown Handler
class GracefulShutdown:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.tasks_to_complete = 0
        self.lock = threading.Lock()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def add_task(self):
        with self.lock:
            self.tasks_to_complete += 1

    def task_complete(self):
        with self.lock:
            self.tasks_to_complete -= 1

    def shutdown(self):
        self.shutdown_event.set()

        start_time = time.time()
        while self.tasks_to_complete > 0 and time.time() - start_time < 30:  # 30 seconds timeout
            time.sleep(0.1)

        if self.tasks_to_complete > 0:
            logger.warning(f"{self.tasks_to_complete} tasks did not complete before shutdown")

    def is_shutting_down(self):
        return self.shutdown_event.is_set()

graceful_shutdown = GracefulShutdown()

# Database connection handling
@contextmanager
def get_db_connection(timeout=5):
    connection = None
    try:
        connection = connection_pool.getconn(key=None, timeout=timeout)
        yield connection
    except psycopg2.pool.PoolError:
        logger.error("Unable to get a connection from the pool within the specified timeout.")
        raise
    finally:
        if connection:
            connection_pool.putconn(connection)

@contextmanager
def get_db_cursor(commit=False):
    with get_db_connection() as connection:
        cursor = connection.cursor()
        try:
            yield cursor
            if commit:
                connection.commit()
        finally:
            cursor.close()

def execute_query(query, params=None, fetch=True, commit=False):
    with get_db_cursor(commit=commit) as cursor:
        cursor.execute(query, params)
        if fetch:
            return cursor.fetchall()

# Retry decorator
def with_retry(max_attempts=3, base_wait=1, max_wait=10):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=base_wait, max=max_wait),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )

# Circuit Breaker for Tika
tika_breaker = CircuitBreaker(fail_max=5, reset_timeout=60)

@tika_breaker
def parse_document_with_tika(document):
    tika_url = os.getenv('TIKA_SERVER_URL', 'http://localhost:9998')
    response = requests.put(f"{tika_url}/tika", data=document, headers={'Accept': 'application/json'})
    response.raise_for_status()
    return response.json()

# Profiling decorator
def profile(output_file=None):
    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            retval = prof.runcall(func, *args, **kwargs)
            
            s = io.StringIO()
            ps = pstats.Stats(prof, stream=s).sort_stats('cumulative')
            ps.print_stats()
            
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(s.getvalue())
            else:
                print(s.getvalue())
            return retval
        return wrapper
    return inner

# Memory usage decorator
def profile_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        mem_usage, retval = memory_usage((func, args, kwargs), retval=True, timeout=200, interval=1e-7)
        print(f"Peak memory usage: {max(mem_usage)} MiB")
        return retval
    return wrapper

# Simple monitoring class
class ApplicationMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0

    def log_request(self):
        self.request_count += 1

    def log_error(self):
        self.error_count += 1

    def get_uptime(self):
        return time.time() - self.start_time

    def get_cpu_usage(self):
        return psutil.cpu_percent()

    def get_memory_usage(self):
        return psutil.virtual_memory().percent

    def get_stats(self):
        return {
            "uptime": self.get_uptime(),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "cpu_usage": self.get_cpu_usage(),
            "memory_usage": self.get_memory_usage()
        }

# Initialize the monitor
app_monitor = ApplicationMonitor()

# Sentiment Analysis
@with_retry()
def analyze_sentiment(email_id):
    app_monitor.log_request()
    try:
        query = "SELECT subject, body FROM emails WHERE id = %s"
        result = execute_query(query, (email_id,))
        if not result:
            logger.warning(f"No email found with id {email_id}")
            return None

        subject, body = result[0]
        text = subject + " " + body

        from textblob import TextBlob
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        sentiment_subjectivity = blob.sentiment.subjectivity

        sentiment_threshold = float(os.getenv('SENTIMENT_THRESHOLD', 0.1))
        if sentiment_score > sentiment_threshold:
            sentiment = "Positive"
        elif sentiment_score < -sentiment_threshold:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        result = {
            "sentiment": sentiment,
            "score": sentiment_score,
            "subjectivity": sentiment_subjectivity
        }

        execute_query(
            "UPDATE emails SET sentiment_analysis = %s WHERE id = %s",
            (json.dumps(result), email_id),
            fetch=False,
            commit=True
        )

        return result
    except Exception as e:
        app_monitor.log_error()
        logger.error(f"Error in analyze_sentiment: {e}")
        raise


# Topic Modeling
@profile_memory
def perform_topic_modeling(n_topics=None):
    app_monitor.log_request()
    try:
        if n_topics is None:
            n_topics = int(os.getenv('NUM_TOPICS', 5))

        query = "SELECT id, subject || ' ' || body as content FROM emails"
        emails = execute_query(query)
        
        email_ids, texts = zip(*emails)
        
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(texts)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(doc_term_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topics.append({
                'id': topic_idx,
                'top_words': top_words
            })
        
        email_topics = lda.transform(doc_term_matrix)
        for email_id, topic_dist in zip(email_ids, email_topics):
            main_topic = np.argmax(topic_dist)
            execute_query(
                "UPDATE emails SET main_topic = %s, topic_distribution = %s WHERE id = %s",
                (int(main_topic), topic_dist.tolist(), email_id),
                fetch=False,
                commit=True
            )
        
        return topics
    except Exception as e:
        app_monitor.log_error()
        logger.error(f"Error in perform_topic_modeling: {e}")
        raise

def get_emails_by_topic(topic_id, limit=20):
    app_monitor.log_request()
    try:
        query = """
            SELECT id, subject, sent_date
            FROM emails
            WHERE main_topic = %s
            ORDER BY (topic_distribution->>%s)::float DESC
            LIMIT %s
        """
        return execute_query(query, (topic_id, str(topic_id), limit))
    except Exception as e:
        app_monitor.log_error()
        logger.error(f"Error in get_emails_by_topic: {e}")
        raise

# Relationship Mapping
def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]

def build_relationship_graph(batch_size=1000):
    app_monitor.log_request()
    try:
        total_processed = 0
        while not graceful_shutdown.is_shutting_down():
            query = """
                SELECT id, subject, body 
                FROM emails 
                WHERE id > %s 
                ORDER BY id 
                LIMIT %s
            """
            batch = execute_query(query, (total_processed, batch_size))
            if not batch:
                break
            
            for email_id, subject, body in batch:
                text = subject + " " + body
                entities = extract_entities(text)
                
                for entity in entities:
                    if not G.has_node(entity):
                        G.add_node(entity, weight=1)
                    else:
                        G.nodes[entity]['weight'] += 1

                for i in range(len(entities)):
                    for j in range(i+1, len(entities)):
                        if G.has_edge(entities[i], entities[j]):
                            G[entities[i]][entities[j]]['weight'] += 1
                        else:
                            G.add_edge(entities[i], entities[j], weight=1)

            total_processed += len(batch)
            logger.info(f"Processed {total_processed} emails")

        return len(G.nodes), len(G.edges)
    except Exception as e:
        app_monitor.log_error()
        logger.error(f"Error in build_relationship_graph: {e}")
        raise

def get_relationship_data():
    return json.dumps(nx.node_link_data(G))

def get_most_connected_entities(top_n=10):
    return sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n]

# Report Generation
def generate_report(output_file):
    app_monitor.log_request()
    try:
        topics = perform_topic_modeling()
        sentiment_stats = get_sentiment_statistics()
        top_entities = get_most_connected_entities()
        
        doc = SimpleDocTemplate(output_file, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Email Analysis Report", styles['Title']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Top Topics", styles['Heading2']))
        for topic in topics:
            story.append(Paragraph(f"Topic {topic['id']}: {', '.join(topic['top_words'])}", styles['Normal']))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Sentiment Analysis", styles['Heading2']))
        data = [['Sentiment', 'Count']] + list(sentiment_stats.items())
        t = Table(data)
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                               ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                               ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                               ('FONTSIZE', (0, 0), (-1, 0), 14),
                               ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                               ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                               ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                               ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                               ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                               ('FONTSIZE', (0, 1), (-1, -1), 12),
                               ('TOPPADDING', (0, 1), (-1, -1), 6),
                               ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                               ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
        story.append(t)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Most Connected Entities", styles['Heading2']))
        for entity, connections in top_entities:
            story.append(Paragraph(f"{entity}: {connections} connections", styles['Normal']))

        doc.build(story)
        logger.info(f"Report generated and saved as {output_file}")
    except Exception as e:
        app_monitor.log_error()
        logger.error(f"Error in generate_report: {e}")
        raise


# Data Export
def export_data_csv():
    app_monitor.log_request()
    try:
        query = "SELECT id, sender, recipient, subject, sent_date, sentiment_analysis FROM emails"
        rows = execute_query(query)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Sender', 'Recipient', 'Subject', 'Sent Date', 'Sentiment'])
        for row in rows:
            writer.writerow(row)

        return output.getvalue()
    except Exception as e:
        app_monitor.log_error()
        logger.error(f"Error in export_data_csv: {e}")
        raise

# Semantic Search
def semantic_search(query, top_k=10):
    app_monitor.log_request()
    try:
        query_embedding = model.encode([query])[0]

        query = "SELECT id, subject, embedding FROM emails WHERE embedding IS NOT NULL"
        emails = execute_query(query)

        similarities = []
        for email_id, subject, embedding in emails:
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append((email_id, subject, similarity))

        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    except Exception as e:
        app_monitor.log_error()
        logger.error(f"Error in semantic_search: {e}")
        raise

# Red Flag Detection
red_flags = [
    (r'\b(confidential|secret|classified)\b', 'Contains sensitive information'),
    (r'\b(urgent|immediate|asap)\b', 'Marked as urgent'),
    (r'\b(lawsuit|legal action|sue)\b', 'Mentions legal action'),
    (r'\b(scandal|controversy|expose)\b', 'Potential controversy'),
    (r'\b(off the record|not for publication)\b', 'Off-record information'),
    (r'\bmoney\s+transfer\b', 'Mentions money transfer'),
    (r'\b(bribe|kickback|payoff)\b', 'Potential corruption'),
    (r'\b(insider|stock tip)\b', 'Potential insider trading'),
    (r'\b(whistleblow|leak)\b', 'Potential whistleblowing'),
    (r'\b(cover[\s-]up|hide|conceal)\b', 'Potential cover-up'),
]

def check_red_flags(email_id):
    app_monitor.log_request()
    try:
        query = "SELECT subject, body FROM emails WHERE id = %s"
        result = execute_query(query, (email_id,))
        if not result:
            logger.warning(f"No email found with id {email_id}")
            return []

        subject, body = result[0]
        text = subject + " " + body
        
        flags = []
        for pattern, reason in red_flags:
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(reason)
        
        if flags:
            execute_query(
                "UPDATE emails SET red_flags = %s WHERE id = %s",
                (flags, email_id),
                fetch=False,
                commit=True
            )
        
        return flags
    except Exception as e:
        app_monitor.log_error()
        logger.error(f"Error in check_red_flags: {e}")
        raise

# Data Integrity Check
def check_data_integrity():
    app_monitor.log_request()
    try:
        integrity_issues = []

        queries = [
            ("SELECT COUNT(*) FROM emails WHERE sentiment_analysis IS NULL", "emails without sentiment analysis"),
            ("SELECT COUNT(*) FROM emails WHERE embedding IS NULL", "emails without embeddings"),
            ("SELECT COUNT(*) FROM emails WHERE main_topic IS NULL", "emails without assigned topics"),
            ("SELECT COUNT(*) FROM emails WHERE sent_date > %s", "emails with future dates"),
            ("""
            SELECT COUNT(*) FROM (
                SELECT sender, recipient, subject, sent_date, COUNT(*)
                FROM emails
                GROUP BY sender, recipient, subject, sent_date
                HAVING COUNT(*) > 1
            ) AS duplicates
            """, "potential duplicate emails")
        ]

        for query, issue_description in queries:
            params = (datetime.now(),) if "sent_date > %s" in query else None
            result = execute_query(query, params)[0][0]
            if result > 0:
                integrity_issues.append(f"{result} {issue_description}")

        return integrity_issues
    except Exception as e:
        app_monitor.log_error()
        logger.error(f"Error in check_data_integrity: {e}")
        raise

# Scheduler for periodic tasks
def schedule_integrity_check():
    schedule.every().day.at("02:00").do(run_integrity_check)

def run_integrity_check():
    logger.info("Running scheduled data integrity check")
    issues = check_data_integrity()
    if issues:
        logger.warning(f"Data integrity issues found: {', '.join(issues)}")
        # You might want to implement fix_data_integrity_issues() here
    else:
        logger.info("No data integrity issues found")

# Shutdown handler
def shutdown_handler(signum, frame):
    logger.info(f"Received shutdown signal {signum}. Initiating graceful shutdown...")
    graceful_shutdown.shutdown()

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

def close_resources():
    logger.info("Closing resources...")
    
    if connection_pool:
        connection_pool.closeall()
        logger.info("Closed all database connections.")

    # Stop the scheduler if it's running
    schedule.clear()
    logger.info("Stopped the scheduler.")

    logger.info("All resources closed.")

atexit.register(close_resources)

# Gradio interface
def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Email Analyzer for Journalists")

        with gr.Tab("Sentiment Analysis"):
            email_id_input = gr.Number(label="Email ID")
            analyze_sentiment_button = gr.Button("Analyze Sentiment")
            sentiment_output = gr.Textbox(label="Sentiment Analysis Result")
            analyze_sentiment_button.click(analyze_sentiment, inputs=email_id_input, outputs=sentiment_output)

        with gr.Tab("Topic Modeling"):
            n_topics_input = gr.Slider(minimum=2, maximum=20, step=1, label="Number of Topics", value=5)
            perform_topic_modeling_button = gr.Button("Perform Topic Modeling")
            topic_modeling_output = gr.Textbox(label="Topics")
            perform_topic_modeling_button.click(perform_topic_modeling, inputs=n_topics_input, outputs=topic_modeling_output)

        with gr.Tab("Relationship Mapping"):
            build_graph_button = gr.Button("Build Relationship Graph")
            build_graph_output = gr.Textbox(label="Graph Building Result")
            build_graph_button.click(build_relationship_graph, outputs=build_graph_output)

        with gr.Tab("Report Generation"):
            report_output_file = gr.Textbox(label="Report Output File Name (e.g., report.pdf)")
            generate_report_button = gr.Button("Generate Report")
            report_output = gr.Textbox(label="Report Generation Result")
            generate_report_button.click(generate_report, inputs=report_output_file, outputs=report_output)

        with gr.Tab("Data Export"):
            export_csv_button = gr.Button("Export Data as CSV")
            csv_output = gr.File(label="Exported CSV Data")
            export_csv_button.click(export_data_csv, outputs=csv_output)

        with gr.Tab("Semantic Search"):
            search_query = gr.Textbox(label="Search Query")
            search_button = gr.Button("Perform Semantic Search")
            search_results = gr.Textbox(label="Search Results")
            search_button.click(semantic_search, inputs=search_query, outputs=search_results)

        with gr.Tab("Red Flag Detection"):
            red_flag_email_id = gr.Number(label="Email ID")
            check_red_flags_button = gr.Button("Check Red Flags")
            red_flags_output = gr.Textbox(label="Red Flags Result")
            check_red_flags_button.click(check_red_flags, inputs=red_flag_email_id, outputs=red_flags_output)

        with gr.Tab("Data Integrity"):
            check_integrity_button = gr.Button("Check Data Integrity")
            integrity_output = gr.Textbox(label="Integrity Check Result")
            check_integrity_button.click(check_data_integrity, outputs=integrity_output)

        with gr.Tab("Application Monitor"):
            monitor_button = gr.Button("Get Application Stats")
            monitor_output = gr.JSON(label="Application Stats")
            monitor_button.click(lambda: app_monitor.get_stats(), outputs=monitor_output)

    return demo

if __name__ == "__main__":
    schedule_integrity_check()
    
    # Start scheduler in a separate thread
    scheduler_thread = threading.Thread(target=lambda: schedule.run_pending(), daemon=True)
    scheduler_thread.start()

    # Create and launch Gradio interface
    demo = create_gradio_interface()
    demo.launch()