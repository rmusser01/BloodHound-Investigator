# Gradio interface
#
#1st-Party Imports
#
#
# Local Imports
from Postgres.App_Function_Libraries.Bloodhound_Investigator_Backend import (analyze_sentiment, perform_topic_modeling, build_relationship_graph, generate_report, export_data_csv, semantic_search, check_red_flags, check_data_integrity, app_monitor)
#
# Third-Party Imports
import gradio as gr
##############################################


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
