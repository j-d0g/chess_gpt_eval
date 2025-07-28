#!/usr/bin/env python3
"""
Enhanced Chess Analysis Dashboard
Combines interactive Stockfish analysis with superior visualization features
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from collections import defaultdict

class EnhancedAnalysisDashboard:
    def __init__(self, results_dir='data/analysis'):
        self.results_dir = Path(results_dir)
        self.app = dash.Dash(__name__)
        self.load_data()
        self.setup_layout()
        self.setup_callbacks()
        
    def load_data(self):
        """Load all analysis data with enhanced processing"""
        self.summary_data = {}
        self.moves_data = {}
        self.model_list = []
        
        # Define required columns for valid analysis files
        required_columns = {
            'average_centipawn_loss', 'blunders', 'mistakes', 'inaccuracies',
            'best_moves', 'good_moves', 'suboptimal_moves', 'opening_accuracy',
            'middlegame_accuracy', 'endgame_accuracy'
        }
        
        # Load summary data
        for file in self.results_dir.glob('*_summary_*.csv'):
            if '202506' in file.name or '202507' in file.name:  # Recent analyses
                model_name = file.name.split('_vs_')[0]
                try:
                    df = pd.read_csv(file)
                    if required_columns.issubset(df.columns):
                        self.summary_data[model_name] = df
                        if model_name not in self.model_list:
                            self.model_list.append(model_name)
                    else:
                        print(f"WARNING: Skipping incomplete file: {file.name}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        # Load moves data if available
        for file in self.results_dir.glob('*_moves_*.csv'):
            if '202506' in file.name or '202507' in file.name:
                model_name = file.name.split('_vs_')[0]
                try:
                    df = pd.read_csv(file)
                    self.moves_data[model_name] = df
                except Exception as e:
                    print(f"Error loading moves data {file}: {e}")
        
        self.model_list = sorted(list(set(self.model_list)))
        
        # Prepare enhanced statistics
        self.prepare_enhanced_stats()
    
    def prepare_enhanced_stats(self):
        """Prepare enhanced aggregate statistics"""
        stats = []
        for model, df in self.summary_data.items():
            if len(df) > 0:
                # Calculate enhanced metrics
                consistency = 1 / (df['average_centipawn_loss'].std() + 1)
                
                # Estimate Elo based on performance
                elo_estimate = self.estimate_elo(df)
                
                # Calculate phase strengths
                phase_balance = (df['opening_accuracy'].mean() + 
                               df['middlegame_accuracy'].mean() + 
                               df['endgame_accuracy'].mean()) / 3
                
                stats.append({
                    'model': model,
                    'games': len(df),
                    'avg_cp_loss': df['average_centipawn_loss'].mean(),
                    'avg_blunders': df['blunders'].mean(),
                    'avg_mistakes': df['mistakes'].mean(),
                    'avg_inaccuracies': df['inaccuracies'].mean(),
                    'opening_acc': df['opening_accuracy'].mean(),
                    'middlegame_acc': df['middlegame_accuracy'].mean(),
                    'endgame_acc': df['endgame_accuracy'].mean(),
                    'consistency': consistency,
                    'elo_estimate': elo_estimate,
                    'phase_balance': phase_balance,
                    'tactical_strength': df['best_moves'].mean() + df['good_moves'].mean(),
                    'error_rate': df['blunders'].mean() + df['mistakes'].mean()
                })
        
        self.aggregate_stats = pd.DataFrame(stats)
    
    def estimate_elo(self, df):
        """Estimate Elo rating from centipawn loss"""
        # Rough mapping: lower CP loss = higher Elo
        avg_cp_loss = df['average_centipawn_loss'].mean()
        
        # Empirical mapping (can be refined)
        if avg_cp_loss <= 50:
            return 1800 + (50 - avg_cp_loss) * 8
        elif avg_cp_loss <= 100:
            return 1600 + (100 - avg_cp_loss) * 4
        elif avg_cp_loss <= 200:
            return 1400 + (200 - avg_cp_loss) * 2
        else:
            return max(1200, 1400 - (avg_cp_loss - 200))
    
    def setup_layout(self):
        """Setup enhanced dashboard layout"""
        self.app.layout = html.Div([
            html.Div([
                html.H1("Enhanced Chess Analysis Dashboard", 
                       style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2c3e50'}),
                
                html.Div([
                    html.Div([
                        html.Label("Select Models to Compare:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='model-selector',
                            options=[{'label': m.replace('_pt', '').replace('_', ' '), 
                                    'value': m} for m in self.model_list],
                            value=self.model_list[:4] if self.model_list else None,
                            multi=True,
                            style={'marginBottom': 10}
                        )
                    ], style={'width': '45%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Analysis View:", style={'fontWeight': 'bold'}),
                        dcc.RadioItems(
                            id='analysis-type',
                            options=[
                                {'label': 'Performance Overview', 'value': 'overview'},
                                {'label': 'Move Quality Analysis', 'value': 'move_quality'},
                                {'label': 'Game Phase Analysis', 'value': 'phases'},
                                {'label': 'Error Pattern Analysis', 'value': 'errors'},
                                {'label': 'Comparative Analysis', 'value': 'comparison'}
                            ],
                            value='overview',
                            inline=True,
                            style={'marginTop': 10}
                        )
                    ], style={'width': '50%', 'float': 'right', 'display': 'inline-block'})
                ], style={'marginBottom': 30}),
                
                # Enhanced controls
                html.Div([
                    html.Div([
                        html.Label("Metric Focus:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='metric-focus',
                            options=[
                                {'label': 'Centipawn Loss', 'value': 'cp_loss'},
                                {'label': 'Move Quality', 'value': 'move_quality'},
                                {'label': 'Tactical Strength', 'value': 'tactical'},
                                {'label': 'Positional Play', 'value': 'positional'},
                                {'label': 'Consistency', 'value': 'consistency'}
                            ],
                            value='cp_loss',
                            style={'marginBottom': 10}
                        )
                    ], style={'width': '30%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Game Phase Filter:", style={'fontWeight': 'bold'}),
                        dcc.Checklist(
                            id='phase-filter',
                            options=[
                                {'label': 'Opening', 'value': 'opening'},
                                {'label': 'Middlegame', 'value': 'middlegame'},
                                {'label': 'Endgame', 'value': 'endgame'}
                            ],
                            value=['opening', 'middlegame', 'endgame'],
                            inline=True
                        )
                    ], style={'width': '35%', 'display': 'inline-block', 'marginLeft': '5%'}),
                    
                    html.Div([
                        html.Label("Show Confidence Intervals:", style={'fontWeight': 'bold'}),
                        dcc.Checklist(
                            id='show-confidence',
                            options=[{'label': 'Enable', 'value': 'enabled'}],
                            value=[],
                            inline=True
                        )
                    ], style={'width': '25%', 'float': 'right', 'display': 'inline-block'})
                ], style={'marginBottom': 30, 'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px'}),
                
                # Main content area
                html.Div(id='main-content', style={'marginBottom': 30}),
                
                # Enhanced summary section
                html.Div([
                    html.H3("Model Performance Summary", style={'color': '#2c3e50'}),
                    html.Div(id='summary-table'),
                    
                    html.H3("Key Insights", style={'color': '#2c3e50', 'marginTop': 30}),
                    html.Div(id='insights-panel')
                ], style={'marginTop': 40})
                
            ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})
        ])
    
    def setup_callbacks(self):
        """Setup enhanced dashboard callbacks"""
        
        @self.app.callback(
            [Output('main-content', 'children'),
             Output('summary-table', 'children'),
             Output('insights-panel', 'children')],
            [Input('model-selector', 'value'),
             Input('analysis-type', 'value'),
             Input('metric-focus', 'value'),
             Input('phase-filter', 'value'),
             Input('show-confidence', 'value')]
        )
        def update_dashboard(selected_models, analysis_type, metric_focus, 
                           phase_filter, show_confidence):
            if not selected_models:
                return (html.Div("Please select at least one model.", 
                               style={'textAlign': 'center', 'color': '#e74c3c'}), 
                       html.Div([]), html.Div([]))
            
            # Filter models to only those with data
            models_with_data = [m for m in selected_models if m in self.aggregate_stats['model'].values]
            if not models_with_data:
                return (html.Div("No valid data available for the selected models.", 
                               style={'textAlign': 'center', 'color': '#e74c3c'}), 
                       html.Div([]), html.Div([]))
            
            show_ci = 'enabled' in show_confidence
            
            # Generate main content based on analysis type
            if analysis_type == 'overview':
                main_content = self.create_overview_analysis(models_with_data, metric_focus, show_ci)
            elif analysis_type == 'move_quality':
                main_content = self.create_move_quality_analysis(models_with_data, show_ci)
            elif analysis_type == 'phases':
                main_content = self.create_phase_analysis(models_with_data, phase_filter, show_ci)
            elif analysis_type == 'errors':
                main_content = self.create_error_analysis(models_with_data, show_ci)
            else:
                main_content = self.create_comparative_analysis(models_with_data, metric_focus, show_ci)
            
            # Create enhanced summary table
            summary_table = self.create_enhanced_summary_table(models_with_data)
            
            # Generate insights
            insights = self.generate_insights(models_with_data, analysis_type)
            
            return main_content, summary_table, insights
    
    def create_overview_analysis(self, models, metric_focus, show_ci):
        """Create enhanced overview analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Comparison', 'Elo Estimates',
                          'Phase Performance', 'Error Distribution'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "violin"}]]
        )
        
        # 1. Performance comparison
        for i, model in enumerate(models):
            model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
            
            if metric_focus == 'cp_loss':
                y_val = model_data['avg_cp_loss']
                y_title = 'Average Centipawn Loss'
            elif metric_focus == 'consistency':
                y_val = model_data['consistency']
                y_title = 'Consistency Score'
            else:
                y_val = model_data['tactical_strength']
                y_title = 'Tactical Strength'
            
            fig.add_trace(
                go.Scatter(
                    x=[model.replace('_pt', '').replace('_', ' ')],
                    y=[y_val],
                    mode='markers',
                    marker=dict(size=15, color=px.colors.qualitative.Set3[i % 10]),
                    name=model.replace('_pt', '').replace('_', ' ')
                ),
                row=1, col=1
            )
        
        # 2. Elo estimates
        for i, model in enumerate(models):
            model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
            fig.add_trace(
                go.Bar(
                    x=[model.replace('_pt', '').replace('_', ' ')],
                    y=[model_data['elo_estimate']],
                    name=model.replace('_pt', '').replace('_', ' '),
                    marker_color=px.colors.qualitative.Set3[i % 10],
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Phase performance
        phases = ['opening_acc', 'middlegame_acc', 'endgame_acc']
        phase_names = ['Opening', 'Middlegame', 'Endgame']
        
        for i, model in enumerate(models):
            model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
            
            fig.add_trace(
                go.Bar(
                    x=phase_names,
                    y=[model_data[phase] for phase in phases],
                    name=model.replace('_pt', '').replace('_', ' '),
                    marker_color=px.colors.qualitative.Set3[i % 10],
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Error distribution
        for i, model in enumerate(models):
            if model in self.summary_data:
                df = self.summary_data[model]
                fig.add_trace(
                    go.Violin(
                        y=df['average_centipawn_loss'],
                        name=model.replace('_pt', '').replace('_', ' '),
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=px.colors.qualitative.Set3[i % 10],
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=800,
            title_text="Enhanced Performance Overview",
            showlegend=True
        )
        
        # Update axis titles
        fig.update_yaxes(title_text=y_title, row=1, col=1)
        fig.update_yaxes(title_text="Estimated Elo", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy %", row=2, col=1)
        fig.update_yaxes(title_text="Centipawn Loss", row=2, col=2)
        
        return dcc.Graph(figure=fig)
    
    def create_move_quality_analysis(self, models, show_ci):
        """Create enhanced move quality analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Move Quality Distribution', 'Blunder Rate Comparison',
                          'Best Move Percentage', 'Quality vs Consistency'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Move quality distribution (stacked bar)
        quality_categories = ['best_moves', 'good_moves', 'suboptimal_moves', 'inaccuracies', 'mistakes', 'blunders']
        quality_names = ['Best', 'Good', 'Suboptimal', 'Inaccuracies', 'Mistakes', 'Blunders']
        colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
        
        for i, category in enumerate(quality_categories):
            y_values = []
            for model in models:
                model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
                if model in self.summary_data:
                    avg_val = self.summary_data[model][category].mean()
                    y_values.append(avg_val)
                else:
                    y_values.append(0)
            
            fig.add_trace(
                go.Bar(
                    x=[m.replace('_pt', '').replace('_', ' ') for m in models],
                    y=y_values,
                    name=quality_names[i],
                    marker_color=colors[i]
                ),
                row=1, col=1
            )
        
        # 2. Blunder rate comparison
        for i, model in enumerate(models):
            model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
            
            fig.add_trace(
                go.Scatter(
                    x=[model.replace('_pt', '').replace('_', ' ')],
                    y=[model_data['avg_blunders']],
                    mode='markers',
                    marker=dict(size=model_data['games']/50, color=px.colors.qualitative.Set3[i % 10]),
                    name=model.replace('_pt', '').replace('_', ' '),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Best move percentage
        for i, model in enumerate(models):
            if model in self.summary_data:
                df = self.summary_data[model]
                best_pct = df['best_moves'].mean()
                
                fig.add_trace(
                    go.Bar(
                        x=[model.replace('_pt', '').replace('_', ' ')],
                        y=[best_pct],
                        name=model.replace('_pt', '').replace('_', ' '),
                        marker_color=px.colors.qualitative.Set3[i % 10],
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 4. Quality vs Consistency scatter
        for i, model in enumerate(models):
            model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
            
            fig.add_trace(
                go.Scatter(
                    x=[model_data['tactical_strength']],
                    y=[model_data['consistency']],
                    mode='markers+text',
                    marker=dict(size=15, color=px.colors.qualitative.Set3[i % 10]),
                    text=[model.replace('_pt', '').replace('_', ' ')],
                    textposition="top center",
                    name=model.replace('_pt', '').replace('_', ' '),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Move Quality Analysis",
            barmode='stack'
        )
        
        # Update axis titles
        fig.update_yaxes(title_text="Average Moves per Game", row=1, col=1)
        fig.update_yaxes(title_text="Blunders per Game", row=1, col=2)
        fig.update_yaxes(title_text="Best Moves per Game", row=2, col=1)
        fig.update_yaxes(title_text="Consistency Score", row=2, col=2)
        fig.update_xaxes(title_text="Tactical Strength", row=2, col=2)
        
        return dcc.Graph(figure=fig)
    
    def create_phase_analysis(self, models, phase_filter, show_ci):
        """Create enhanced phase analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Phase Performance Comparison', 'Phase Balance',
                          'Opening vs Endgame', 'Phase Consistency'),
            specs=[[{"type": "bar"}, {"type": "scatterpolar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Phase performance comparison
        phases = []
        phase_names = []
        if 'opening' in phase_filter:
            phases.append('opening_acc')
            phase_names.append('Opening')
        if 'middlegame' in phase_filter:
            phases.append('middlegame_acc')
            phase_names.append('Middlegame')
        if 'endgame' in phase_filter:
            phases.append('endgame_acc')
            phase_names.append('Endgame')
        
        for i, phase in enumerate(phases):
            y_values = []
            for model in models:
                model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
                y_values.append(model_data[phase])
            
            fig.add_trace(
                go.Bar(
                    x=[m.replace('_pt', '').replace('_', ' ') for m in models],
                    y=y_values,
                    name=phase_names[i],
                    marker_color=px.colors.qualitative.Set3[i % 10]
                ),
                row=1, col=1
            )
        
        # 2. Phase balance (radar chart)
        for i, model in enumerate(models):
            model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=[model_data['opening_acc'], model_data['middlegame_acc'], model_data['endgame_acc']],
                    theta=['Opening', 'Middlegame', 'Endgame'],
                    fill='toself',
                    name=model.replace('_pt', '').replace('_', ' '),
                    line_color=px.colors.qualitative.Set3[i % 10]
                ),
                row=1, col=2
            )
        
        # 3. Opening vs Endgame scatter
        for i, model in enumerate(models):
            model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
            
            fig.add_trace(
                go.Scatter(
                    x=[model_data['opening_acc']],
                    y=[model_data['endgame_acc']],
                    mode='markers+text',
                    marker=dict(size=15, color=px.colors.qualitative.Set3[i % 10]),
                    text=[model.replace('_pt', '').replace('_', ' ')],
                    textposition="top center",
                    name=model.replace('_pt', '').replace('_', ' '),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Phase consistency
        for i, model in enumerate(models):
            model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
            phase_std = np.std([model_data['opening_acc'], model_data['middlegame_acc'], model_data['endgame_acc']])
            
            fig.add_trace(
                go.Bar(
                    x=[model.replace('_pt', '').replace('_', ' ')],
                    y=[1 / (phase_std + 0.1)],  # Higher is more consistent
                    name=model.replace('_pt', '').replace('_', ' '),
                    marker_color=px.colors.qualitative.Set3[i % 10],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Game Phase Analysis"
        )
        
        # Update axis titles
        fig.update_yaxes(title_text="Accuracy %", row=1, col=1)
        fig.update_yaxes(title_text="Endgame Accuracy", row=2, col=1)
        fig.update_xaxes(title_text="Opening Accuracy", row=2, col=1)
        fig.update_yaxes(title_text="Phase Consistency", row=2, col=2)
        
        return dcc.Graph(figure=fig)
    
    def create_error_analysis(self, models, show_ci):
        """Create enhanced error analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Error Type Distribution', 'Error Rate vs Performance',
                          'Blunder Frequency', 'Error Correlation'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "heatmap"}]]
        )
        
        # 1. Error type distribution
        error_types = ['blunders', 'mistakes', 'inaccuracies']
        error_names = ['Blunders', 'Mistakes', 'Inaccuracies']
        colors = ['#e74c3c', '#f39c12', '#f1c40f']
        
        for i, error_type in enumerate(error_types):
            y_values = []
            for model in models:
                model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
                y_values.append(model_data[f'avg_{error_type}'])
            
            fig.add_trace(
                go.Bar(
                    x=[m.replace('_pt', '').replace('_', ' ') for m in models],
                    y=y_values,
                    name=error_names[i],
                    marker_color=colors[i]
                ),
                row=1, col=1
            )
        
        # 2. Error rate vs Performance
        for i, model in enumerate(models):
            model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
            
            fig.add_trace(
                go.Scatter(
                    x=[model_data['error_rate']],
                    y=[model_data['elo_estimate']],
                    mode='markers+text',
                    marker=dict(size=15, color=px.colors.qualitative.Set3[i % 10]),
                    text=[model.replace('_pt', '').replace('_', ' ')],
                    textposition="top center",
                    name=model.replace('_pt', '').replace('_', ' '),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Blunder frequency histogram
        for i, model in enumerate(models):
            if model in self.summary_data:
                df = self.summary_data[model]
                fig.add_trace(
                    go.Histogram(
                        x=df['blunders'],
                        name=model.replace('_pt', '').replace('_', ' '),
                        opacity=0.7,
                        marker_color=px.colors.qualitative.Set3[i % 10]
                    ),
                    row=2, col=1
                )
        
        # 4. Error correlation heatmap
        if len(models) >= 2:
            corr_data = []
            for model in models[:4]:  # Limit to 4 for readability
                model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
                corr_data.append([
                    model_data['avg_blunders'],
                    model_data['avg_mistakes'],
                    model_data['avg_inaccuracies'],
                    model_data['avg_cp_loss']
                ])
            
            corr_matrix = np.corrcoef(corr_data)
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix,
                    x=[m.replace('_pt', '').replace('_', ' ') for m in models[:4]],
                    y=[m.replace('_pt', '').replace('_', ' ') for m in models[:4]],
                    colorscale='RdBu',
                    zmid=0,
                    showscale=True
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Error Pattern Analysis",
            barmode='group'
        )
        
        # Update axis titles
        fig.update_yaxes(title_text="Average per Game", row=1, col=1)
        fig.update_yaxes(title_text="Estimated Elo", row=1, col=2)
        fig.update_xaxes(title_text="Error Rate", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Blunders per Game", row=2, col=1)
        
        return dcc.Graph(figure=fig)
    
    def create_comparative_analysis(self, models, metric_focus, show_ci):
        """Create enhanced comparative analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Rankings', 'Performance Distribution',
                          'Strength vs Consistency', 'Improvement Potential'),
            specs=[[{"type": "bar"}, {"type": "violin"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Model rankings
        rankings = []
        for model in models:
            model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
            rankings.append((model, model_data['elo_estimate']))
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        fig.add_trace(
            go.Bar(
                x=[r[0].replace('_pt', '').replace('_', ' ') for r in rankings],
                y=[r[1] for r in rankings],
                marker_color=px.colors.qualitative.Set3[:len(rankings)],
                name='Elo Ranking'
            ),
            row=1, col=1
        )
        
        # 2. Performance distribution
        for i, model in enumerate(models):
            if model in self.summary_data:
                df = self.summary_data[model]
                fig.add_trace(
                    go.Violin(
                        y=df['average_centipawn_loss'],
                        name=model.replace('_pt', '').replace('_', ' '),
                        box_visible=True,
                        meanline_visible=True,
                        fillcolor=px.colors.qualitative.Set3[i % 10],
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. Strength vs Consistency
        for i, model in enumerate(models):
            model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
            
            fig.add_trace(
                go.Scatter(
                    x=[model_data['tactical_strength']],
                    y=[model_data['consistency']],
                    mode='markers+text',
                    marker=dict(
                        size=model_data['games']/50,
                        color=model_data['elo_estimate'],
                        colorscale='viridis',
                        showscale=True,
                        colorbar=dict(title="Elo")
                    ),
                    text=[model.replace('_pt', '').replace('_', ' ')],
                    textposition="top center",
                    name=model.replace('_pt', '').replace('_', ' '),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. Improvement potential
        for i, model in enumerate(models):
            model_data = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
            
            # Calculate improvement potential (inverse of error rate)
            improvement_score = 1 / (model_data['error_rate'] + 0.1)
            
            fig.add_trace(
                go.Bar(
                    x=[model.replace('_pt', '').replace('_', ' ')],
                    y=[improvement_score],
                    name=model.replace('_pt', '').replace('_', ' '),
                    marker_color=px.colors.qualitative.Set3[i % 10],
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Comparative Analysis"
        )
        
        # Update axis titles
        fig.update_yaxes(title_text="Estimated Elo", row=1, col=1)
        fig.update_yaxes(title_text="Centipawn Loss", row=1, col=2)
        fig.update_yaxes(title_text="Consistency", row=2, col=1)
        fig.update_xaxes(title_text="Tactical Strength", row=2, col=1)
        fig.update_yaxes(title_text="Improvement Potential", row=2, col=2)
        
        return dcc.Graph(figure=fig)
    
    def create_enhanced_summary_table(self, models):
        """Create enhanced summary table with more metrics"""
        table_data = []
        
        for model in models:
            if model in self.aggregate_stats['model'].values:
                stats = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
                table_data.append({
                    'Model': model.replace('_pt', '').replace('_', ' '),
                    'Games': f"{stats['games']:,}",
                    'Elo Est.': f"{stats['elo_estimate']:.0f}",
                    'Avg CP Loss': f"{stats['avg_cp_loss']:.1f}",
                    'Blunders/Game': f"{stats['avg_blunders']:.2f}",
                    'Opening Acc': f"{stats['opening_acc']:.1f}%",
                    'Middlegame Acc': f"{stats['middlegame_acc']:.1f}%",
                    'Endgame Acc': f"{stats['endgame_acc']:.1f}%",
                    'Consistency': f"{stats['consistency']:.2f}",
                    'Tactical Strength': f"{stats['tactical_strength']:.1f}"
                })
        
        # Sort by Elo estimate
        table_data.sort(key=lambda x: float(x['Elo Est.']), reverse=True)
        
        if table_data:
            return dash_table.DataTable(
                data=table_data,
                columns=[{"name": i, "id": i} for i in table_data[0].keys()],
                style_cell={'textAlign': 'center', 'fontSize': 12},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                sort_action="native",
                filter_action="native"
            )
        
        return html.Div("No data available for selected models")
    
    def generate_insights(self, models, analysis_type):
        """Generate intelligent insights based on the data"""
        insights = []
        
        if not models:
            return html.Div("No models selected for analysis.")
        
        # Get model data
        model_data = []
        for model in models:
            if model in self.aggregate_stats['model'].values:
                stats = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
                model_data.append(stats)
        
        if not model_data:
            return html.Div("No data available for insight generation.")
        
        # Best performer
        best_model = max(model_data, key=lambda x: x['elo_estimate'])
        insights.append(
            html.Div([
                html.H4("🏆 Best Performer", style={'color': '#27ae60'}),
                html.P(f"{best_model['model'].replace('_', ' ')} leads with an estimated Elo of {best_model['elo_estimate']:.0f} "
                      f"and {best_model['avg_cp_loss']:.1f} average centipawn loss.")
            ])
        )
        
        # Most consistent
        most_consistent = max(model_data, key=lambda x: x['consistency'])
        insights.append(
            html.Div([
                html.H4("🎯 Most Consistent", style={'color': '#3498db'}),
                html.P(f"{most_consistent['model'].replace('_', ' ')} shows the highest consistency with a score of {most_consistent['consistency']:.2f}.")
            ])
        )
        
        # Phase specialist
        opening_specialist = max(model_data, key=lambda x: x['opening_acc'])
        endgame_specialist = max(model_data, key=lambda x: x['endgame_acc'])
        
        insights.append(
            html.Div([
                html.H4("♟️ Phase Specialists", style={'color': '#e67e22'}),
                html.P(f"Opening: {opening_specialist['model'].replace('_', ' ')} ({opening_specialist['opening_acc']:.1f}%)"),
                html.P(f"Endgame: {endgame_specialist['model'].replace('_', ' ')} ({endgame_specialist['endgame_acc']:.1f}%)")
            ])
        )
        
        # Improvement opportunity
        highest_errors = max(model_data, key=lambda x: x['error_rate'])
        insights.append(
            html.Div([
                html.H4("📈 Improvement Opportunity", style={'color': '#e74c3c'}),
                html.P(f"{highest_errors['model'].replace('_', ' ')} has the highest error rate ({highest_errors['error_rate']:.2f}) "
                      f"and could benefit from tactical training.")
            ])
        )
        
        return html.Div(insights, style={'backgroundColor': '#f8f9fa', 'padding': '15px', 'borderRadius': '5px'})
    
    def run(self, debug=False, port=8050):
        """Run the enhanced dashboard"""
        print("Starting Enhanced Chess Analysis Dashboard...")
        print(f"Dashboard will be available at: http://localhost:{port}")
        print("Features:")
        print("- Enhanced performance analysis")
        print("- Interactive move quality analysis")
        print("- Game phase breakdown")
        print("- Error pattern analysis")
        print("- Comparative model analysis")
        print("- Intelligent insights generation")
        
        self.app.run_server(debug=debug, port=port)


if __name__ == "__main__":
    dashboard = EnhancedAnalysisDashboard()
    dashboard.run(debug=True) 