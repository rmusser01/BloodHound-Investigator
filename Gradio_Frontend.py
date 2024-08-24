import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import networkx as nx
import json
import os
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, conint, ValidationError

# Import functions from the backend
from email_analyzer_backend import (
    analyze_sentiment,
    perform_topic_modeling,
    get_emails_by_topic,
    build_relationship_graph,
    get_relationship_data,
    get_most_connected_entities,
    generate_report,
    export_data_csv,
    semantic_search,
    check_red_flags,
    batch_check_red_flags,
    get_red_flagged_emails,
    classify_and_store_source,
    batch_classify_sources,
    get_emails_by_classification,
    get_classification_statistics,
    close_db_connections
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Input validation models
class EmailID(BaseModel):
    email_id: conint(gt=0)

class TopicModel(BaseModel):
    n_topics: conint(gt=1, lt=100)

class SearchQuery(BaseModel):
    query: str

# Gradio interface functions with error handling and input validation
def analyze_sentiment_interface(email_id):
    try:
        EmailID(email_id=email_id)
        result = analyze_sentiment(email_id)
        return f"Sentiment: {result['sentiment']}\nScore: {result['score']:.2f}\nSubjectivity: {result['subjectivity']:.2f}"
    except ValidationError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Error in analyze_sentiment_interface: {e}")
        return f"An error occurred: {str(e)}"

def perform_topic_modeling_interface(n_topics):
    try:
        TopicModel(n_topics=n_topics)
        topics = perform_topic_modeling(n_topics=n_topics)
        return "\n".join([f"Topic {t['id']}: {', '.join(t['top_words'])}" for t in topics])
    except ValidationError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Error in perform_topic_modeling_interface: {e}")
        return f"An error occurred: {str(e)}"

def get_emails_by_topic_interface(topic_id):
    try:
        EmailID(email_id=topic_id)  # Reusing EmailID for topic_id validation
        emails = get_emails_by_topic(topic_id)
        return "\n\n".join([f"ID: {e[0]}, Subject: {e[1]}, Date: {e[2]}" for e in emails])
    except ValidationError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Error in get_emails_by_topic_interface: {e}")
        return f"An error occurred: {str(e)}"

def build_relationship_graph_interface():
    try:
        nodes, edges = build_relationship_graph()
        return f"Built relationship graph with {nodes} nodes and {edges} edges"
    except Exception as e:
        logger.error(f"Error in build_relationship_graph_interface: {e}")
        return f"An error occurred: {str(e)}"

def visualize_relationships_interface():
    try:
        graph_data = json.loads(get_relationship_data())
        G = nx.node_link_graph(graph_data)

        pos = nx.spring_layout(G)
        
        edge_trace = go.Scatter(
            x=[], y=[], line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        node_trace = go.Scatter(
            x=[], y=[], text=[], mode='markers', hoverinfo='text',
            marker=dict(
                showscale=True, colorscale='YlGnBu', reversescale=True, color=[], size=10,
                colorbar=dict(thickness=15, title='Node Connections', xanchor='left', titleside='right'),
                line_width=2))

        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])

        for node, adjacencies in enumerate(G.adjacency()):
            node_info = G.nodes[adjacencies[0]]
            node_trace['marker']['color'] += tuple([len(adjacencies[1])])
            node_info_text = f"{adjacencies[0]}<br># of connections: {len(adjacencies[1])}"
            node_trace['text'] += tuple([node_info_text])

        fig = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                        title='Entity Relationship Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(text="", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig
    except Exception as e:
        logger.error(f"Error in visualize_relationships_interface: {e}")
        return None

def get_most_connected_entities_interface():
    try:
        entities = get_most_connected_entities()
        return "\n".join([f"{entity}: {connections} connections" for entity, connections in entities])
    except Exception as e:
        logger.error(f"Error in get_most_connected_entities_interface: {e}")
        return f"An error occurred: {str(e)}"

def generate_report_interface(output_file):
    try:
        generate_report(output_file)
        return f"Report generated and saved as {output_file}"
    except Exception as e:
        logger.error(f"Error in generate_report_interface: {e}")
        return f"An error occurred: {str(e)}"

def export_data_csv_interface():
    try:
        csv_data = export_data_csv()
        return csv_data
    except Exception as e:
        logger.error(f"Error in export_data_csv_interface: {e}")
        return f"An error occurred: {str(e)}"

def perform_semantic_search_interface(query):
    try:
        SearchQuery(query=query)
        results = semantic_search(query)
        return "\n\n".join([f"ID: {r[0]}, Subject: {r[1]}, Similarity: {r[2]:.4f}" for r in results])
    except ValidationError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Error in perform_semantic_search_interface: {e}")
        return f"An error occurred: {str(e)}"

def check_red_flags_interface(email_id):
    try:
        EmailID(email_id=email_id)
        flags = check_red_flags(email_id)
        if flags:
            return f"Red flags found: {', '.join(flags)}"
        else:
            return "No red flags found."
    except ValidationError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Error in check_red_flags_interface: {e}")
        return f"An error occurred: {str(e)}"

def batch_check_red_flags_interface():
    try:
        count = batch_check_red_flags()
        return f"Checked all emails. Found {count} emails with red flags."
    except Exception as e:
        logger.error(f"Error in batch_check_red_flags_interface: {e}")
        return f"An error occurred: {str(e)}"

def get_red_flagged_emails_interface():
    try:
        emails = get_red_flagged_emails()
        return "\n\n".join([f"ID: {e[0]}, Subject: {e[1]}, Date: {e[2]}, Flags: {', '.join(e[3])}" for e in emails])
    except Exception as e:
        logger.error(f"Error in get_red_flagged_emails_interface: {e}")
        return f"An error occurred: {str(e)}"

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Email Analyzer for Journalists")

    with gr.Tab("Sentiment Analysis"):
        email_id_input = gr.Number(label="Email ID")
        analyze_sentiment_button = gr.Button("Analyze Sentiment")
        sentiment_output = gr.Textbox(label="Sentiment Analysis Result")
        analyze_sentiment_button.click(analyze_sentiment_interface, inputs=email_id_input, outputs=sentiment_output)

    with gr.Tab("Topic Modeling"):
        n_topics_input = gr.Slider(minimum=2, maximum=20, step=1, label="Number of Topics", value=5)
        perform_topic_modeling_button = gr.Button("Perform Topic Modeling")
        topic_modeling_output = gr.Textbox(label="Topics")
        perform_topic_modeling_button.click(perform_topic_modeling_interface, inputs=n_topics_input, outputs=topic_modeling_output)

        gr.Markdown("### View Emails by Topic")
        topic_id_input = gr.Number(label="Topic ID")
        view_topic_button = gr.Button("View Emails in Topic")
        topic_emails_output = gr.Textbox(label="Emails in Topic")
        view_topic_button.click(get_emails_by_topic_interface, inputs=topic_id_input, outputs=topic_emails_output)

    with gr.Tab("Relationship Mapping"):
        build_graph_button = gr.Button("Build Relationship Graph")
        build_graph_output = gr.Textbox(label="Graph Building Result")
        build_graph_button.click(build_relationship_graph_interface, outputs=build_graph_output)

        visualize_button = gr.Button("Visualize Relationships")
        relationship_graph = gr.Plot(label="Relationship Graph")
        visualize_button.click(visualize_relationships_interface, outputs=relationship_graph)

        most_connected_button = gr.Button("Show Most Connected Entities")
        most_connected_output = gr.Textbox(label="Most Connected Entities")
        most_connected_button.click(get_most_connected_entities_interface, outputs=most_connected_output)

    with gr.Tab("Export and Report"):
        report_output_file = gr.Textbox(label="Report Output File Name (e.g., report.pdf)")
        generate_report_button = gr.Button("Generate Report")
        report_output = gr.Textbox(label="Report Generation Result")
        generate_report_button.click(generate_report_interface, inputs=report_output_file, outputs=report_output)

        export_csv_button = gr.Button("Export Data as CSV")
        csv_output = gr.File(label="Exported CSV Data")
        export_csv_button.click(export_data_csv_interface, outputs=csv_output)

    with gr.Tab("Semantic Search"):
        search_query = gr.Textbox(label="Search Query")
        search_button = gr.Button("Perform Semantic Search")
        search_results = gr.Textbox(label="Search Results")
        search_button.click(perform_semantic_search_interface, inputs=search_query, outputs=search_results)

    with gr.Tab("Red Flag Tagging"):
        email_id_input = gr.Number(label="Email ID")
        check_flags_button = gr.Button("Check Red Flags")
        flags_output = gr.Textbox(label="Red Flags Result")
        check_flags_button.click(check_red_flags_interface, inputs=email_id_input, outputs=flags_output)

        batch_check_button = gr.Button("Batch Check All Emails")
        batch_check_output = gr.Textbox(label="Batch Check Result")
        batch_check_button.click(batch_check_red_flags_interface, outputs=batch_check_output)

        view_flagged_button = gr.Button("View Red Flagged Emails")
        flagged_emails_output = gr.Textbox(label="Red Flagged Emails")
        view_flagged_button.click(get_red_flagged_emails_interface, outputs=flagged_emails_output)

if __name__ == "__main__":
    try:
        demo.launch()
    finally:
        close_db_connections()