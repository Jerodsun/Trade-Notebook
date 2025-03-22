import os
import json
import datetime as dt
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import sqlite3
import pandas as pd


# Database functions for annotations
def create_annotations_table(db_path="spx_data.db"):
    """Create annotations table if it doesn't exist"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create annotations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS annotations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        datetime TEXT,
        price REAL,
        text TEXT,
        arrow_direction TEXT,
        color TEXT,
        created_at TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

def save_annotation(annotation, db_path="spx_data.db"):
    """Save a new annotation to the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Extract annotation data
    date = annotation.get('date')
    datetime_str = annotation.get('datetime')
    price = annotation.get('price')
    text = annotation.get('text')
    arrow_direction = annotation.get('arrow_direction', 'up')
    color = annotation.get('color', 'red')
    created_at = dt.datetime.now().isoformat()
    
    # Insert annotation into database
    cursor.execute('''
    INSERT INTO annotations (date, datetime, price, text, arrow_direction, color, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (date, datetime_str, price, text, arrow_direction, color, created_at))
    
    # Get the ID of the inserted annotation
    annotation_id = cursor.lastrowid
    
    conn.commit()
    conn.close()
    
    return annotation_id

def get_annotations_for_date(date, db_path="spx_data.db"):
    """Get all annotations for a specific date"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, date, datetime, price, text, arrow_direction, color, created_at
    FROM annotations
    WHERE date = ?
    ORDER BY datetime
    ''', (date,))
    
    # Fetch all annotations
    annotations = []
    for row in cursor.fetchall():
        annotations.append({
            'id': row[0],
            'date': row[1],
            'datetime': row[2],
            'price': row[3],
            'text': row[4],
            'arrow_direction': row[5],
            'color': row[6],
            'created_at': row[7]
        })
    
    conn.close()
    return annotations

def get_annotations_for_range(start_date, end_date, db_path="spx_data.db"):
    """Get all annotations for a date range"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, date, datetime, price, text, arrow_direction, color, created_at
    FROM annotations
    WHERE date >= ? AND date <= ?
    ORDER BY datetime
    ''', (start_date, end_date))
    
    # Fetch all annotations
    annotations = []
    for row in cursor.fetchall():
        annotations.append({
            'id': row[0],
            'date': row[1],
            'datetime': row[2],
            'price': row[3],
            'text': row[4],
            'arrow_direction': row[5],
            'color': row[6],
            'created_at': row[7]
        })
    
    conn.close()
    return annotations

def delete_annotation(annotation_id, db_path="spx_data.db"):
    """Delete an annotation from the database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM annotations WHERE id = ?', (annotation_id,))
    
    conn.commit()
    conn.close()
    
    return True

def update_annotation(annotation, db_path="spx_data.db"):
    """Update an existing annotation"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Extract annotation data
    annotation_id = annotation.get('id')
    text = annotation.get('text')
    color = annotation.get('color')
    arrow_direction = annotation.get('arrow_direction')
    
    # Update annotation in database
    cursor.execute('''
    UPDATE annotations
    SET text = ?, color = ?, arrow_direction = ?
    WHERE id = ?
    ''', (text, color, arrow_direction, annotation_id))
    
    conn.commit()
    conn.close()
    
    return True

# Function to add annotations to a figure
def add_annotations_to_figure(fig, annotations, y_domain=[0, 1]):
    """Add annotations to a plotly figure"""
    if not annotations:
        return fig
    
    for annotation in annotations:
        # Parse datetime and price
        datetime_str = annotation.get('datetime')
        price = annotation.get('price')
        text = annotation.get('text')
        color = annotation.get('color', 'red')
        arrow_direction = annotation.get('arrow_direction', 'up')
        
        # Set arrow properties based on direction
        if arrow_direction == 'up':
            ay = -40
            arrowhead = 2
        elif arrow_direction == 'down':
            ay = 40
            arrowhead = 2
        else:  # None or other
            ay = 0
            arrowhead = 0
        
        # Add the annotation to the figure
        fig.add_annotation(
            x=datetime_str,
            y=price,
            text=text,
            showarrow=True,
            arrowhead=arrowhead,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color,
            ax=0,
            ay=ay,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor=color,
            borderwidth=2,
            borderpad=4,
            font=dict(color=color, size=12)
        )
    
    return fig

# Annotation UI components
def create_annotation_modal():
    """Create a modal dialog for adding/editing annotations"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Add Chart Annotation")),
            dbc.ModalBody([
                dbc.Form([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Annotation Text:"),
                            dbc.Textarea(
                                id="annotation-text",
                                placeholder="Enter your annotation...",
                                rows=3
                            ),
                        ], width=12),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Color:"),
                            dbc.Select(
                                id="annotation-color",
                                options=[
                                    {"label": "Red", "value": "red"},
                                    {"label": "Green", "value": "green"},
                                    {"label": "Blue", "value": "blue"},
                                    {"label": "Orange", "value": "orange"},
                                    {"label": "Purple", "value": "purple"},
                                ],
                                value="red",
                            ),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Arrow Direction:"),
                            dbc.Select(
                                id="annotation-arrow",
                                options=[
                                    {"label": "Up", "value": "up"},
                                    {"label": "Down", "value": "down"},
                                    {"label": "None", "value": "none"},
                                ],
                                value="up",
                            ),
                        ], width=6),
                    ]),
                    html.Br(),
                    dbc.Row([
                        dbc.Col([
                            html.P("Click on the chart to place the annotation.", className="text-muted"),
                            html.Div(id="annotation-coordinates", className="text-muted"),
                        ], width=12),
                    ]),
                ]),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="cancel-annotation-button", className="ms-auto", color="secondary"),
                dbc.Button("Save", id="save-annotation-button", color="primary"),
            ]),
        ],
        id="annotation-modal",
        centered=True,
    )

def create_annotation_manager():
    """Create a component for managing annotations"""
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Span("Annotations", className="h5 me-2"),
                dbc.Button(
                    html.I(className="fas fa-plus"),
                    id="add-annotation-button",
                    color="primary",
                    size="sm",
                    className="me-2",
                ),
                dbc.Button(
                    html.I(className="fas fa-sync-alt"),
                    id="refresh-annotations-button",
                    color="secondary",
                    size="sm",
                    className="me-2",
                ),
            ], className="d-flex align-items-center"),
        ]),
        dbc.CardBody([
            html.Div(
                id="annotations-list",
                style={"maxHeight": "300px", "overflowY": "auto"},
                children=[]
            ),
        ]),
    ])

def create_annotation_list_item(annotation):
    """Create a list item for an annotation"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.P(f"{annotation['text']}", className="mb-0 fw-bold"),
                    html.Small(
                        f"At: {annotation['datetime']} - Price: {annotation['price']}",
                        className="text-muted"
                    ),
                ], className="me-auto"),
                html.Div([
                    dbc.Button(
                        html.I(className="fas fa-pencil-alt"),
                        id={"type": "edit-annotation", "index": annotation["id"]},
                        color="link",
                        size="sm",
                        className="p-0 me-2",
                    ),
                    dbc.Button(
                        html.I(className="fas fa-trash"),
                        id={"type": "delete-annotation", "index": annotation["id"]},
                        color="link",
                        size="sm",
                        className="p-0 text-danger",
                    ),
                ], className="d-flex"),
            ], className="d-flex justify-content-between align-items-center"),
        ], className="p-2")
    ], className="mb-2", style={"borderLeft": f"4px solid {annotation['color']}"})