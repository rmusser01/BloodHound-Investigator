# Gradio interface
#
#1st-Party Imports
from datetime import datetime

import plotly.graph_objs as go
import networkx as nx
import json
#
#
# Local Imports
from Postgres.App_Function_Libraries.Bloodhound_Investigator_Backend import (analyze_sentiment, perform_topic_modeling,
                                                                             build_relationship_graph, generate_report,
                                                                             export_data_csv, semantic_search,
                                                                             check_red_flags, check_data_integrity,
                                                                             app_monitor, logger, get_emails_by_topic,
                                                                             get_relationship_data,
                                                                             get_most_connected_entities, error_handler,
                                                                             validate_input_decorator, get_email_by_id,
                                                                             analyze_email, analyze_email_thread,
                                                                             analyze_sentiment_trend)
#
# Third-Party Imports
import gradio as gr
##############################################

# Gradio interface functions with error handling and input validation

# Input validation models (you might want to move these to a separate file)
from pydantic import BaseModel, conint, constr, ValidationError

class EmailID(BaseModel):
    email_id: conint(gt=0)

class TopicModel(BaseModel):
    n_topics: conint(gt=1, lt=100)

class SearchQuery(BaseModel):
    query: constr(min_length=1, max_length=200)

# Utility function for input validation
def validate_input(input_data, model):
    try:
        validated_data = model(**input_data)
        return validated_data
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        raise ValueError(str(e))


@error_handler
@validate_input_decorator(EmailID)
def analyze_sentiment_interface(email_id):
    try:
        validate_input({"email_id": email_id}, EmailID)
        result = analyze_sentiment(email_id)
        return f"Sentiment: {result['sentiment']}\nScore: {result['score']:.2f}\nSubjectivity: {result['subjectivity']:.2f}"
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Error in analyze_sentiment_interface: {e}")
        return f"An error occurred: {str(e)}"


@error_handler
@validate_input_decorator(EmailID)
def perform_topic_modeling_interface(n_topics):
    try:
        validate_input({"n_topics": n_topics}, TopicModel)
        topics = perform_topic_modeling(n_topics=n_topics)
        return "\n".join([f"Topic {t['id']}: {', '.join(t['top_words'])}" for t in topics])
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Error in perform_topic_modeling_interface: {e}")
        return f"An error occurred: {str(e)}"


@error_handler
@validate_input_decorator(EmailID)
def get_emails_by_topic_interface(topic_id):
    try:
        validate_input({"email_id": topic_id}, EmailID)  # Reusing EmailID for topic_id validation
        emails = get_emails_by_topic(topic_id)
        return "\n\n".join([f"ID: {e[0]}, Subject: {e[1]}, Date: {e[2]}" for e in emails])
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Error in get_emails_by_topic_interface: {e}")
        return f"An error occurred: {str(e)}"


@error_handler
def build_relationship_graph_interface():
    try:
        G, nodes, edges = build_relationship_graph()
        # Store G in a global variable or in a session state
        global relationship_graph
        relationship_graph = G
        return f"Built relationship graph with {nodes} nodes and {edges} edges"
    except Exception as e:
        logger.error(f"Error in build_relationship_graph_interface: {e}")
        return f"An error occurred: {str(e)}"


@error_handler
def visualize_relationships_interface():
    try:
        global relationship_graph
        if relationship_graph is None:
            return "Please build the relationship graph first."

        graph_data = json.loads(get_relationship_data(relationship_graph))
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
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(text="", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
        return fig
    except Exception as e:
        logger.error(f"Error in visualize_relationships_interface: {e}")
        return None


@error_handler
def get_most_connected_entities_interface():
    try:
        global relationship_graph
        if relationship_graph is None:
            return "Please build the relationship graph first."

        entities = get_most_connected_entities(relationship_graph)
        return "\n".join([f"{entity}: {connections} connections" for entity, connections in entities])
    except Exception as e:
        logger.error(f"Error in get_most_connected_entities_interface: {e}")
        return f"An error occurred: {str(e)}"


@error_handler
def generate_report_interface(output_file):
    try:
        generate_report(output_file)
        return f"Report generated and saved as {output_file}"
    except Exception as e:
        logger.error(f"Error in generate_report_interface: {e}")
        return f"An error occurred: {str(e)}"


@error_handler
def export_data_csv_interface():
    try:
        csv_data = export_data_csv()
        return csv_data
    except Exception as e:
        logger.error(f"Error in export_data_csv_interface: {e}")
        return f"An error occurred: {str(e)}"


@error_handler
def perform_semantic_search_interface(query):
    try:
        validate_input({"query": query}, SearchQuery)
        results = semantic_search(query)
        return "\n\n".join([f"ID: {r[0]}, Subject: {r[1]}, Similarity: {r[2]:.4f}" for r in results])
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Error in perform_semantic_search_interface: {e}")
        return f"An error occurred: {str(e)}"


@error_handler
def check_red_flags_interface(email_id):
    try:
        validate_input({"email_id": email_id}, EmailID)
        flags = check_red_flags(email_id)
        if flags:
            return f"Red flags found: {', '.join(flags)}"
        else:
            return "No red flags found."
    except ValueError as e:
        return f"Invalid input: {str(e)}"
    except Exception as e:
        logger.error(f"Error in check_red_flags_interface: {e}")
        return f"An error occurred: {str(e)}"


@error_handler
def check_data_integrity_interface():
    try:
        issues = check_data_integrity()
        if issues:
            return "Data integrity issues found:\n" + "\n".join(issues)
        else:
            return "No data integrity issues found."
    except Exception as e:
        logger.error(f"Error in check_data_integrity_interface: {e}")
        return f"An error occurred: {str(e)}"


@error_handler
def get_application_stats_interface():
    try:
        stats = app_monitor.get_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:
        logger.error(f"Error in get_application_stats_interface: {e}")
        return f"An error occurred: {str(e)}"


@error_handler
@validate_input_decorator(EmailID)
def analyze_email_interface(email_id):
    email = get_email_by_id(email_id)
    if not email:
        raise ValueError(f"No email found with id {email_id}")
    return analyze_email(email)

@error_handler
@validate_input_decorator(EmailID)
def analyze_thread_interface(thread_id):
    return analyze_email_thread(thread_id)

@error_handler
def analyze_sentiment_trend_interface(start_date, end_date):
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.")
    return analyze_sentiment_trend(start, end)


# Gradio interface
def create_gradio_interface():
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
            perform_topic_modeling_button.click(perform_topic_modeling_interface, inputs=n_topics_input,
                                                outputs=topic_modeling_output)

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

        with gr.Tab("Report Generation"):
            report_output_file = gr.Textbox(label="Report Output File Name (e.g., report.pdf)")
            generate_report_button = gr.Button("Generate Report")
            report_output = gr.Textbox(label="Report Generation Result")
            generate_report_button.click(generate_report_interface, inputs=report_output_file, outputs=report_output)

        with gr.Tab("Data Export"):
            export_csv_button = gr.Button("Export Data as CSV")
            csv_output = gr.File(label="Exported CSV Data")
            export_csv_button.click(export_data_csv_interface, outputs=csv_output)

        with gr.Tab("Semantic Search"):
            search_query = gr.Textbox(label="Search Query")
            search_button = gr.Button("Perform Semantic Search")
            search_results = gr.Textbox(label="Search Results")
            search_button.click(perform_semantic_search_interface, inputs=search_query, outputs=search_results)

        with gr.Tab("Red Flag Detection"):
            red_flag_email_id = gr.Number(label="Email ID")
            check_red_flags_button = gr.Button("Check Red Flags")
            red_flags_output = gr.Textbox(label="Red Flags Result")
            check_red_flags_button.click(check_red_flags_interface, inputs=red_flag_email_id, outputs=red_flags_output)

        with gr.Tab("Data Integrity"):
            check_integrity_button = gr.Button("Check Data Integrity")
            integrity_output = gr.Textbox(label="Integrity Check Result")
            check_integrity_button.click(check_data_integrity_interface, outputs=integrity_output)

        with gr.Tab("Application Monitor"):
            monitor_button = gr.Button("Get Application Stats")
            monitor_output = gr.JSON(label="Application Stats")
            monitor_button.click(get_application_stats_interface, outputs=monitor_output)

        with gr.Tab("Advanced Analysis"):
            email_id_input = gr.Number(label="Email ID")
            analyze_button = gr.Button("Analyze Email")
            analysis_output = gr.JSON(label="Analysis Result")

            thread_id_input = gr.Number(label="Thread ID")
            thread_analyze_button = gr.Button("Analyze Thread")
            thread_analysis_output = gr.JSON(label="Thread Analysis Result")

            start_date_input = gr.Textbox(label="Start Date (YYYY-MM-DD)")
            end_date_input = gr.Textbox(label="End Date (YYYY-MM-DD)")
            sentiment_trend_button = gr.Button("Analyze Sentiment Trend")
            sentiment_trend_output = gr.JSON(label="Sentiment Trend Result")

            analyze_button.click(analyze_email_interface, inputs=email_id_input, outputs=analysis_output)
            thread_analyze_button.click(analyze_thread_interface, inputs=thread_id_input, outputs=thread_analysis_output)
            sentiment_trend_button.click(analyze_sentiment_trend_interface, inputs=[start_date_input, end_date_input],
                                         outputs=sentiment_trend_output)

    return demo