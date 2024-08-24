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

# Database connection pool
db_pool = pool.SimpleConnectionPool(
    1,
    20,
    database=os.getenv('DB_NAME', 'email_analyzer'),
    user=os.getenv('DB_USER', 'postgres'),
    password=os.getenv('DB_PASSWORD', 'postgres'),
    host=os.getenv('DB_HOST', 'localhost')
)

def get_db_connection():
    return db_pool.getconn()

def return_db_connection(conn):
    db_pool.putconn(conn)

def execute_query(query, params=None, fetch=True):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Database error: {e}")
        conn.rollback()
        raise
    finally:
        return_db_connection(conn)

def analyze_sentiment(email_id):
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
            fetch=False
        )

        return result
    except Exception as e:
        logger.error(f"Error in analyze_sentiment: {e}")
        raise

def perform_topic_modeling(n_topics=None):
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
                fetch=False
            )
        
        return topics
    except Exception as e:
        logger.error(f"Error in perform_topic_modeling: {e}")
        raise

def get_emails_by_topic(topic_id, limit=20):
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
        logger.error(f"Error in get_emails_by_topic: {e}")
        raise

def extract_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']]

def build_relationship_graph(batch_size=1000):
    try:
        total_processed = 0
        while True:
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
        logger.error(f"Error in build_relationship_graph: {e}")
        raise

def get_relationship_data():
    return json.dumps(nx.node_link_data(G))

def get_most_connected_entities(top_n=10):
    return sorted(G.degree, key=lambda x: x[1], reverse=True)[:top_n]

def generate_report(output_file):
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
        logger.error(f"Error in generate_report: {e}")
        raise

def export_data_csv():
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
        logger.error(f"Error in export_data_csv: {e}")
        raise

def get_sentiment_statistics():
    try:
        query = """
            SELECT 
                sentiment_analysis->>'sentiment' as sentiment,
                COUNT(*) as count
            FROM emails
            GROUP BY sentiment_analysis->>'sentiment'
        """
        return dict(execute_query(query))
    except Exception as e:
        logger.error(f"Error in get_sentiment_statistics: {e}")
        raise

def semantic_search(query, top_k=10):
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
        logger.error(f"Error in semantic_search: {e}")
        raise

# Red flag patterns
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
                fetch=False
            )
        
        return flags
    except Exception as e:
        logger.error(f"Error in check_red_flags: {e}")
        raise

def batch_check_red_flags(batch_size=1000):
    try:
        total_processed = 0
        flagged_count = 0
        while True:
            query = """
                SELECT id, subject, body 
                FROM emails 
                WHERE id > %s AND red_flags IS NULL
                ORDER BY id 
                LIMIT %s
            """
            batch = execute_query(query, (total_processed, batch_size))
            if not batch:
                break
            
            for email_id, subject, body in batch:
                text = subject + " " + body
                flags = []
                for pattern, reason in red_flags:
                    if re.search(pattern, text, re.IGNORECASE):
                        flags.append(reason)
                
                if flags:
                    execute_query(
                        "UPDATE emails SET red_flags = %s WHERE id = %s",
                        (flags, email_id),
                        fetch=False
                    )
                    flagged_count += 1
            
            total_processed += len(batch)
            logger.info(f"Processed {total_processed} emails, found {flagged_count} with red flags")
        
        return flagged_count
    except Exception as e:
        logger.error(f"Error in batch_check_red_flags: {e}")
        raise

def get_red_flagged_emails(limit=20):
    try:
        query = """
            SELECT id, subject, sent_date, red_flags
            FROM emails
            WHERE red_flags IS NOT NULL
            ORDER BY sent_date DESC
            LIMIT %s
        """
        return execute_query(query, (limit,))
    except Exception as e:
        logger.error(f"Error in get_red_flagged_emails: {e}")
        raise

# Domain classification patterns
domain_classifications = {
    r'\.gov$': 'Government Official',
    r'\.edu$': 'Academic',
    r'\.(com|org)$': 'Corporate/NGO',
    r'\.mil$': 'Military',
    r'\.media$': 'Media',
    r'^pr@': 'PR Representative',
}

def classify_source(email):
    sender = email.get('from', '').lower()
    domain = sender.split('@')[-1] if '@' in sender else ''

    for pattern, classification in domain_classifications.items():
        if re.search(pattern, sender):
            return classification

    return 'Other'

def classify_and_store_source(email_id):
    try:
        query = "SELECT sender FROM emails WHERE id = %s"
        result = execute_query(query, (email_id,))
        if not result:
            logger.warning(f"No email found with id {email_id}")
            return None

        sender = result[0][0]
        classification = classify_source({'from': sender})
        
        execute_query(
            "UPDATE emails SET source_classification = %s WHERE id = %s",
            (classification, email_id),
            fetch=False
        )
        
        return classification
    except Exception as e:
        logger.error(f"Error in classify_and_store_source: {e}")
        raise

def batch_classify_sources(batch_size=1000):
    try:
        total_processed = 0
        while True:
            query = """
                SELECT id, sender 
                FROM emails 
                WHERE id > %s AND source_classification IS NULL
                ORDER BY id 
                LIMIT %s
            """
            batch = execute_query(query, (total_processed, batch_size))
            if not batch:
                break
            
            for email_id, sender in batch:
                classification = classify_source({'from': sender})
                execute_query(
                    "UPDATE emails SET source_classification = %s WHERE id = %s",
                    (classification, email_id),
                    fetch=False
                )
            
            total_processed += len(batch)
            logger.info(f"Classified sources for {total_processed} emails")
        
        return total_processed
    except Exception as e:
        logger.error(f"Error in batch_classify_sources: {e}")
        raise

def get_emails_by_classification(classification, limit=20):
    try:
        query = """
            SELECT id, sender, subject, sent_date
            FROM emails
            WHERE source_classification = %s
            ORDER BY sent_date DESC
            LIMIT %s
        """
        return execute_query(query, (classification, limit))
    except Exception as e:
        logger.error(f"Error in get_emails_by_classification: {e}")
        raise

def get_classification_statistics():
    try:
        query = """
            SELECT source_classification, COUNT(*) as count
            FROM emails
            GROUP BY source_classification
            ORDER BY count DESC
        """
        return execute_query(query)
    except Exception as e:
        logger.error(f"Error in get_classification_statistics: {e}")
        raise

def close_db_connections():
    db_pool.closeall()
    logger.info("Closed all database connections.")