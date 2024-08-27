import os
import subprocess
import sys
import time
import venv
import logging
from datetime import datetime

import psycopg2
import threading
import schedule
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command):
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Error executing command: {command}")
            logger.error(stderr.decode())
            raise Exception(f"Command failed: {command}")
        return stdout.decode()
    except Exception as e:
        logger.error(f"An error occurred while running command: {command}")
        logger.error(str(e))
        sys.exit(1)

def create_virtual_environment():
    logger.info("Creating virtual environment...")
    venv.create('venv', with_pip=True)
    logger.info("Virtual environment created successfully.")

def run_in_virtual_env(command):
    activate_this = os.path.join('venv', 'bin', 'activate_this.py')
    exec(open(activate_this).read(), {'__file__': activate_this})
    return run_command(command)

def setup_database():
    logger.info("Setting up database...")
    try:
        conn = psycopg2.connect(dbname="postgres", user="postgres", password="postgres", host="localhost")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        cur.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'email_analyzer'")
        exists = cur.fetchone()
        if not exists:
            cur.execute("CREATE DATABASE email_analyzer")
            logger.info("Database 'email_analyzer' created.")
        else:
            logger.info("Database 'email_analyzer' already exists.")

        cur.close()
        conn.close()

        conn = psycopg2.connect(dbname="email_analyzer", user="postgres", password="postgres", host="localhost")
        cur = conn.cursor()

        # Create emails table with partitioning
        cur.execute("""
        CREATE TABLE IF NOT EXISTS emails_partitioned (
            id SERIAL,
            sender VARCHAR(255),
            recipient VARCHAR(255),
            subject TEXT,
            body TEXT,
            sent_date TIMESTAMP,
            attachment BYTEA,
            attachment_name VARCHAR(255),
            embedding FLOAT[],
            sentiment_analysis JSONB,
            main_topic INTEGER,
            topic_distribution FLOAT[],
            thread_id INTEGER,
            in_reply_to INTEGER,
            source_classification TEXT,
            red_flags TEXT[],
            entity_tags TEXT[]
        ) PARTITION BY RANGE (sent_date)
        """)

        # Create partitions
        cur.execute("""
        CREATE TABLE IF NOT EXISTS emails_y2021 PARTITION OF emails_partitioned
            FOR VALUES FROM ('2021-01-01') TO ('2022-01-01')
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS emails_y2022 PARTITION OF emails_partitioned
            FOR VALUES FROM ('2022-01-01') TO ('2023-01-01')
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS emails_y2023 PARTITION OF emails_partitioned
            FOR VALUES FROM ('2023-01-01') TO ('2024-01-01')
        """)

        # Rename tables if necessary
        cur.execute("ALTER TABLE IF EXISTS emails RENAME TO emails_old")
        cur.execute("ALTER TABLE emails_partitioned RENAME TO emails")

        # Create email_analysis table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS email_analysis (
            email_id INTEGER PRIMARY KEY REFERENCES emails(id),
            sentiment_score FLOAT,
            sentiment_label VARCHAR(10),
            entities JSONB,
            topics JSONB,
            keywords JSONB,
            importance_score FLOAT,
            analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_sent_date ON emails (sent_date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_sender ON emails (sender)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_recipient ON emails (recipient)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_thread_id ON emails (thread_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_in_reply_to ON emails (in_reply_to)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_main_topic ON emails (main_topic)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_source_classification ON emails (source_classification)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_red_flags ON emails USING GIN (red_flags)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_entity_tags ON emails USING GIN (entity_tags)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_text_search ON emails USING GIN (to_tsvector('english', subject || ' ' || body))")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email_analysis_email_id ON email_analysis (email_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email_analysis_sentiment_score ON email_analysis (sentiment_score)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_email_analysis_sentiment_label ON email_analysis (sentiment_label)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_emails_subject_body ON emails USING gin (to_tsvector('english', subject || ' ' || body))")

        # Enable vector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Create function for topic distribution matching
        cur.execute("""
        CREATE OR REPLACE FUNCTION match_topic_distribution(topic_dist FLOAT[], topic_id INTEGER)
        RETURNS FLOAT AS $$
        BEGIN
            RETURN topic_dist[topic_id + 1];
        END;
        $$ LANGUAGE plpgsql
        """)

        conn.commit()
        cur.close()
        conn.close()
        logger.info("Database setup completed successfully.")
    except psycopg2.Error as e:
        logger.error(f"An error occurred while setting up the database: {e}")
        sys.exit(1)

def install_requirements():
    logger.info("Installing required packages...")
    requirements = [
        "psycopg2-binary",
        "sentence-transformers",
        "spacy",
        "scikit-learn",
        "networkx",
        "tika",
        "textblob",
        "reportlab",
        "gradio",
        "plotly",
        "pandas",
        "python-dotenv",
        "pydantic",
        "cachetools"
    ]

    for req in requirements:
        logger.info(f"Installing {req}...")
        run_in_virtual_env(f"pip install {req}")

    logger.info("Downloading spaCy English model...")
    run_in_virtual_env("python -m spacy download en_core_web_sm")

def create_config_file():
    logger.info("Creating configuration file...")
    config = """
DATABASE_URL=postgresql://postgres:postgres@localhost/email_analyzer
TIKA_SERVER_URL=http://localhost:9998
SENTIMENT_THRESHOLD=0.1
NUM_TOPICS=5
    """
    with open(".env", "w") as f:
        f.write(config.strip())
    logger.info("Configuration file created successfully.")

def setup_tika():
    logger.info("Setting up Apache Tika...")
    tika_version = "2.7.0"
    tika_url = f"https://downloads.apache.org/tika/{tika_version}/tika-server-standard-{tika_version}.jar"
    run_command(f"wget {tika_url} -O tika-server.jar")
    logger.info("Apache Tika downloaded successfully.")
    logger.info("To start Tika server, run: java -jar tika-server.jar")

def schedule_integrity_check():
    # Schedule the integrity check to run daily at 2:00 AM
    schedule.every().day.at("02:00").do(run_integrity_check)
    logger.info("Scheduled daily integrity check for 2:00 AM")

    # You can add more scheduled tasks here if needed

    # Run the scheduler in the background
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Wait for 60 seconds before checking again

    # Start the scheduler in a separate thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()

def create_gradio_interface():
    # This function should be implemented in your Gradio UI code
    from Postgres.App_Function_Libraries.Gradio_UI import create_gradio_interface
    return create_gradio_interface()

def run_integrity_check():
    logger.info("Running scheduled data integrity check")
    try:
        # Connect to the database
        conn = psycopg2.connect(dbname="email_analyzer", user="postgres", password="postgres", host="localhost")
        cur = conn.cursor()

        # Check for emails without sentiment analysis
        cur.execute("SELECT COUNT(*) FROM emails WHERE sentiment_analysis IS NULL")
        missing_sentiment = cur.fetchone()[0]

        # Check for emails without embeddings
        cur.execute("SELECT COUNT(*) FROM emails WHERE embedding IS NULL")
        missing_embedding = cur.fetchone()[0]

        # Check for emails with future dates
        cur.execute("SELECT COUNT(*) FROM emails WHERE sent_date > %s", (datetime.now(),))
        future_dates = cur.fetchone()[0]

        # Check for duplicate emails
        cur.execute("""
            SELECT COUNT(*) FROM (
                SELECT sender, recipient, subject, sent_date, COUNT(*)
                FROM emails
                GROUP BY sender, recipient, subject, sent_date
                HAVING COUNT(*) > 1
            ) AS duplicates
        """)
        duplicates = cur.fetchone()[0]

        # Log the results
        if missing_sentiment > 0:
            logger.warning(f"Found {missing_sentiment} emails without sentiment analysis")
        if missing_embedding > 0:
            logger.warning(f"Found {missing_embedding} emails without embeddings")
        if future_dates > 0:
            logger.warning(f"Found {future_dates} emails with future dates")
        if duplicates > 0:
            logger.warning(f"Found {duplicates} potential duplicate emails")

        if missing_sentiment == 0 and missing_embedding == 0 and future_dates == 0 and duplicates == 0:
            logger.info("No data integrity issues found")

        # Close database connection
        cur.close()
        conn.close()

    except Exception as e:
        logger.error(f"Error during integrity check: {str(e)}")


def main():
    logger.info("Starting setup for Email Analyzer application...")

    create_virtual_environment()

    logger.info("Installing required packages...")
    install_requirements()

    logger.info("Setting up database...")
    setup_database()

    logger.info("Creating configuration file...")
    create_config_file()

    logger.info("Setting up Apache Tika...")
    setup_tika()

    logger.info("Setup complete!")

    # Load environment variables
    load_dotenv()

    # Schedule & Run integrity checks
    run_integrity_check()
    #This function performs the actual integrity checks. It:
    #Checks for emails without sentiment analysis
    #Checks for emails without embeddings
    #Checks for emails with future dates
    #Checks for potential duplicate emails
    #Logs warnings for any issues found
    #schedule_integrity_check()

    # Start scheduler in a separate thread
    scheduler_thread = threading.Thread(target=lambda: schedule.run_pending(), daemon=True)
    scheduler_thread.start()

    # Create and launch Gradio interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

if __name__ == "__main__":
    main()