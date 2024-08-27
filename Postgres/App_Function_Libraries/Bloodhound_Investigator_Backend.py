# BloodHound-Investigator: Email Analysis Tool
"""
BloodHound-Investigator: Email Analysis Tool

This module provides functionality for analyzing email data, including
sentiment analysis, topic modeling, and relationship mapping.
"""
import cProfile
############################################################################################################################################################################
# Native Imports
import os
import logging
import pstats
from contextlib import contextmanager
#
# Local Imports
from Postgres.App_Function_Libraries.Gradio_UI import create_gradio_interface
#
# Third-Party Imports
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import tika
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
import json
from reportlab.lib import colors
import spacy
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import csv
import gc
import io
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import asyncio
import schedule
import time
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests
import signal
import threading
import atexit
from memory_profiler import memory_usage
import psutil
from pybreaker import CircuitBreaker
from functools import lru_cache, wraps
#
# End of Imports
############################################################################################################################################################################


############################################################################################################################################################################
#
# Global Variables

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Tika
tika.initVM()

# Global variables

# Setup spacy model
nlp = spacy.load("en_core_web_sm")

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize a networkx graph for relationship mapping
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

#
# End of Global Variables
############################################################################################################################################################################0


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
    with get_db_connection(timeout=5) as connection:
        cursor = connection.cursor()
        try:
            yield cursor
            if commit:
                connection.commit()
        finally:
            cursor.close()


@contextmanager
def db_transaction():
    connection = None
    try:
        connection = get_db_connection(timeout=5)
        yield connection
        connection.commit()
    except Exception as e:
        if connection:
            connection.rollback()
        raise DatabaseError(f"Database transaction failed: {str(e)}")
    finally:
        if connection:
            connection.close()

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

############################################################################################################################################################################
#
# Sentiment Analysis Functions

# FIXME: Implement async version of sentiment analysis
def perform_sentiment_analysis(email_content):
    pass

@with_retry()
def analyze_sentiment(email_id):
    """
    Analyze the sentiment of an email.

    This function performs sentiment analysis on the subject and body of an email.
    It uses the TextBlob library to calculate polarity and subjectivity scores.

    Args:
        email_id (int): The unique identifier of the email to analyze.

    Returns:
        dict: A dictionary containing sentiment analysis results:
            - sentiment (str): 'Positive', 'Negative', or 'Neutral'
            - score (float): The polarity score (-1 to 1)
            - subjectivity (float): The subjectivity score (0 to 1)

    Raises:
        ValueError: If the email_id is not found in the database.
    """
    try:
        # Validate input
        if not isinstance(email_id, int) or email_id <= 0:
            raise InputError(f"Invalid email_id: {email_id}. Must be a positive integer.")

        # Fetch email content
        try:
            email_content = get_email_content(email_id)
        except psycopg2.Error as e:
            logger.error(f"Database error when fetching email {email_id}: {str(e)}")
            raise DatabaseError(f"Failed to fetch email {email_id} from the database.")

        if not email_content:
            raise AnalysisError(f"Email with id {email_id} not found.")

        # Perform sentiment analysis
        try:
            sentiment_result = perform_sentiment_analysis(email_content)
        except Exception as e:
            logger.error(f"Sentiment analysis failed for email {email_id}: {str(e)}")
            raise AnalysisError(f"Sentiment analysis failed for email {email_id}.")

        return sentiment_result

    except EmailAnalyzerError as e:
        # Log the error and re-raise
        logger.error(str(e))
        raise
    except Exception as e:
        # Log unexpected errors
        logger.critical(f"Unexpected error in analyze_sentiment: {str(e)}")
        raise EmailAnalyzerError("An unexpected error occurred during sentiment analysis.")


# Topic Modeling
@profile_memory
def perform_topic_modeling(n_topics=5):
    """
    Perform topic modeling on the email corpus.

    This function uses Latent Dirichlet Allocation (LDA) to identify topics
    in the email corpus. It processes all emails in the database and assigns
    a main topic to each email.

    Args:
        n_topics (int, optional): The number of topics to identify. Defaults to 5.

    Returns:
        list: A list of dictionaries, each representing a topic:
            - id (int): The topic identifier
            - top_words (list): The top 10 words associated with the topic

    Note:
        This function may take a long time to complete for large email corpora.
    """
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

        # Use numpy for faster computations
        topic_word_matrix = np.array(lda.components_)
        top_words_indices = np.argsort(topic_word_matrix, axis=1)[:, -10:]

        topics = []
        for topic_idx, top_indices in enumerate(top_words_indices):
            top_words = [feature_names[i] for i in top_indices]
            topics.append({
                'id': topic_idx,
                'top_words': top_words
            })

        return topics
    except Exception as e:
        app_monitor.log_error()
        logger.error(f"Error in perform_topic_modeling: {e}")
        raise


@lru_cache(maxsize=128)
def get_email_content(email_id):
    query = "SELECT subject, body FROM emails WHERE id = %s"
    return execute_query(query, (email_id,))[0]
#
# End of Sentiment Analysis Functions
############################################################################################################################################################################


############################################################################################################################################################################
#
# Relationship Mapping Functions


# New FIXME
def process_email_for_graph(email_id, subject, body):
    pass


def build_relationship_graph(batch_size=1000):
    try:
        total_processed = 0
        while not graceful_shutdown.is_shutting_down():
            try:
                batch = execute_query(
                    "SELECT id, subject, body FROM emails WHERE id > %s ORDER BY id LIMIT %s",
                    (total_processed, batch_size)
                )
            except psycopg2.Error as e:
                logger.error(f"Database error when fetching email batch: {str(e)}")
                raise DatabaseError("Failed to fetch email batch from the database.")

            if not batch:
                break

            for email_id, subject, body in batch:
                try:
                    process_email_for_graph(email_id, subject, body)
                except Exception as e:
                    logger.error(f"Error processing email {email_id} for graph: {str(e)}")
                    # Continue processing other emails

            total_processed += len(batch)
            logger.info(f"Processed {total_processed} emails")

        return len(G.nodes), len(G.edges)
    except DatabaseError:
        raise
    except Exception as e:
        logger.critical(f"Unexpected error in build_relationship_graph: {str(e)}")
        raise EmailAnalyzerError("An unexpected error occurred while building the relationship graph.")



def get_relationship_data():
    return json.dumps(nx.node_link_data(G))


def get_most_connected_entities(top_n=10):
    return sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n]


#
# End of Relationship Mapping Functions
############################################################################################################################################################################


############################################################################################################################################################################
#
# Analyze Emails Functions

def analyze_email(email):
    pass


def analyze_all_emails():
    for email in process_emails_generator():
        analyze_email(email)


def analyze_email_task(email_id):
    # Perform various analysis tasks
    analyze_sentiment(email_id)
    extract_entities(email_id)
    # ... other analysis functions
    # FIXME


def get_emails_by_topic(topic_id, page=1, per_page=20):
    app_monitor.log_request()
    try:
        offset = (page - 1) * per_page
        query = """
            SELECT id, subject, sent_date
            FROM emails
            WHERE main_topic = %s
            ORDER BY (topic_distribution->>%s)::float DESC
            LIMIT %s OFFSET %s
        """
        return execute_query(query, (topic_id, str(topic_id), per_page, offset))
    except Exception as e:
        app_monitor.log_error()
        logger.error(f"Error in get_emails_by_topic: {e}")
        raise


def process_emails_generator(batch_size=1000):
    total_processed = 0
    while True:
        batch = execute_query(
            "SELECT id, subject, body FROM emails WHERE id > %s ORDER BY id LIMIT %s",
            (total_processed, batch_size)
        )
        if not batch:
            break

        for email in batch:
            yield email

        total_processed += len(batch)


def build_email_threads():
    query = """
    SELECT id, subject, in_reply_to, sent_date
    FROM emails
    ORDER BY sent_date
    """
    emails = execute_query(query)

    threads = {}
    for email_id, subject, in_reply_to, sent_date in emails:
        if in_reply_to and in_reply_to in threads:
            threads[in_reply_to].append(email_id)
        else:
            threads[email_id] = []

    return threads

#
# End of Email Processing Functions
############################################################################################################################################################################


############################################################################################################################################################################
#
# Data import functions

import email
import os
import imaplib
from email.parser import BytesParser
from email.policy import default


def import_from_mbox(mbox_file):
    with open(mbox_file, 'rb') as f:
        messages = BytesParser(policy=default).parse(f)

    for message in messages:
        store_email(message)


def import_from_imap(imap_server, username, password, mailbox='INBOX'):
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(username, password)
    mail.select(mailbox)

    _, message_numbers = mail.search(None, 'ALL')
    for num in message_numbers[0].split():
        _, msg = mail.fetch(num, '(RFC822)')
        email_body = msg[0][1]
        message = email.message_from_bytes(email_body)
        store_email(message)

    mail.close()
    mail.logout()


def import_from_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.eml'):
            with open(os.path.join(directory, filename), 'rb') as f:
                message = BytesParser(policy=default).parse(f)
            store_email(message)


def store_email(message):
    # Extract relevant information from the email message
    sender = message['From']
    recipient = message['To']
    subject = message['Subject']
    date = message['Date']
    body = get_email_body(message)

    # Store in the database
    query = """
    INSERT INTO emails (sender, recipient, subject, body, sent_date)
    VALUES (%s, %s, %s, %s, %s)
    """
    execute_query(query, (sender, recipient, subject, body, date), fetch=False, commit=True)


def get_email_body(message):
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode()
    else:
        return message.get_payload(decode=True).decode()

#
# End of Data Import Functions
############################################################################################################################################################################


############################################################################################################################################################################
#
# Async Functions

async def process_emails_async(emails):
    tasks = [asyncio.create_task(process_single_email(email)) for email in emails]
    await asyncio.gather(*tasks)


# FIXME: This function is not yet implemented
async def analyze_sentiment_async(email):
    pass


# FIXME: This function is not yet implemented
async def extract_entities_async(email):
    pass


async def process_single_email(email):
    # Perform various analysis tasks on a single email
    await analyze_sentiment_async(email)
    await extract_entities_async(email)
    # ... other async processing functions

#
# End of Async Functions
############################################################################################################################################################################



# Relationship Mapping

def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
            entities.append((ent.text, ent.label_))
    return entities


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

# FIXME: This function is not yet implemented
def get_sentiment_statistics():
    pass


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

        emails = execute_query("SELECT id, subject, embedding FROM emails WHERE embedding IS NOT NULL")

        similarities = [
            (email_id, subject, cosine_similarity([query_embedding], [embedding])[0][0])
            for email_id, subject, embedding in emails
        ]

        return sorted(similarities, key=lambda x: x[2], reverse=True)[:top_k]
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
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                flags.append({
                    'reason': reason,
                    'match': match.group(),
                    'start': match.start(),
                    'end': match.end()
                })

        if flags:
            execute_query(
                "UPDATE emails SET red_flags = %s WHERE id = %s",
                (json.dumps(flags), email_id),
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

############################################################################################################################################################################
#
# Report Generation Functions

#
# End of Report Generation Functions
############################################################################################################################################################################


############################################################################################################################################################################
#
# DB Exception Classes

class EmailAnalyzerError(Exception):
    """Base exception for the Email Analyzer application."""

class DatabaseError(EmailAnalyzerError):
    """Exception raised for database-related errors."""

class AnalysisError(EmailAnalyzerError):
    """Exception raised for errors during email analysis."""

class InputError(EmailAnalyzerError):
    """Exception raised for invalid input errors."""

#
# End of DB Exception Classes
############################################################################################################################################################################

############################################################################################################################################################################
#
# Main

if __name__ == "__main__":
    schedule_integrity_check()
    
    # Start scheduler in a separate thread
    scheduler_thread = threading.Thread(target=lambda: schedule.run_pending(), daemon=True)
    scheduler_thread.start()

    # Create and launch Gradio interface
    demo = create_gradio_interface()
    demo.launch()

#
# End of Main
############################################################################################################################################################################