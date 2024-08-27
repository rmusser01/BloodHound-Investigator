import os
import subprocess
import sys
import venv
import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# BloodHound-Investigator-Setup.py

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

        cur.execute("""
        CREATE TABLE IF NOT EXISTS emails (
            id SERIAL PRIMARY KEY,
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
        "pydantic"
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
    logger.info("To run the Email Analyzer application:")
    logger.info("1. Activate the virtual environment: source venv/bin/activate")
    logger.info("2. Start the Tika server: java -jar tika-server.jar")
    logger.info("3. Run the application: python run_email_analyzer.py")

if __name__ == "__main__":
    main()