import logging
import threading

import schedule
from dotenv import load_dotenv
#
# Import functions from the backend
from Postgres.App_Function_Libraries.Gradio_UI import create_gradio_interface
from Postgres.BloodHound_Investigator_Launcher import setup_database, schedule_integrity_check
#
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Main execution
if __name__ == "__main__":
    setup_database()
    schedule_integrity_check()

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
