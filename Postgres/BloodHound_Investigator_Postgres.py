import logging
from dotenv import load_dotenv

# Import functions from the backend
from Postgres.App_Function_Libraries.Bloodhound_Investigator_Backend import (analyze_sentiment, perform_topic_modeling, get_emails_by_topic, build_relationship_graph, get_relationship_data, get_most_connected_entities, generate_report, export_data_csv, semantic_search, check_red_flags, check_data_integrity, app_monitor)
from Postgres.App_Function_Libraries.Gradio_UI import create_gradio_interface

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Main execution
if __name__ == "__main__":
    # Create and launch Gradio interface
    demo = create_gradio_interface()
    demo.launch(
        # Makes the app accessible on the local network
        server_name="0.0.0.0",
        # You can change this port if needed
        server_port=7860,
        # Creates a public link. Set to 'True' if you want it public.
        share=False
    )
