#!/usr/bin/env python3
"""
Advanced Chess Analysis Dashboard - MASSIVELY ENHANCED
Deep granular analysis for understanding chess GPT model learning patterns and performance
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
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

class AdvancedChessAnalysisDashboard:
    def __init__(self, results_dir='data/analysis', 
                 moves_dir='data/analysis'):
        self.results_dir = Path(results_dir)
        self.moves_dir = Path(moves_dir)
        self.app = dash.Dash(__name__)
        self.load_data()
        self.setup_layout()
        self.setup_callbacks()
        
    def load_data(self):
        """Load all analysis data with move-level granularity"""
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
            if '202506' in file.name or '202507' in file.name:
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
        
        # Load move-level data for deep analysis
        for file in self.moves_dir.glob('*_moves_*.csv'):
            if '202506' in file.name:
                model_name = file.name.split('_vs_')[0]
                if model_name in self.model_list:
                    try:
                        # Load sample of moves data (first 50k moves for performance)
                        df = pd.read_csv(file, nrows=50000)
                        self.moves_data[model_name] = df
                        print(f"Loaded {len(df)} moves for {model_name}")
                    except Exception as e:
                        print(f"Error loading moves for {model_name}: {e}")
        
        # Calculate aggregate statistics
        self.calculate_enhanced_stats()
    
    def calculate_enhanced_stats(self):
        """Calculate enhanced statistics with move-level insights"""
        stats = []
        
        for model in self.model_list:
            if model in self.summary_data:
                df = self.summary_data[model]
                
                # Basic stats
                model_stats = {
                    'model': model,
                    'games': len(df),
                    'avg_cp_loss': df['average_centipawn_loss'].mean(),
                    'avg_blunders': df['blunders'].mean(),
                    'avg_mistakes': df['mistakes'].mean(),
                    'opening_acc': df['opening_accuracy'].mean(),
                    'middlegame_acc': df['middlegame_accuracy'].mean(),
                    'endgame_acc': df['endgame_accuracy'].mean()
                }
                
                # Enhanced stats from move-level data
                if model in self.moves_data:
                    moves_df = self.moves_data[model]
                    
                    # Move quality distribution
                    model_stats['best_move_rate'] = (moves_df['move_quality'] == 'best').mean() * 100
                    model_stats['good_move_rate'] = (moves_df['move_quality'] == 'good').mean() * 100
                    model_stats['suboptimal_rate'] = (moves_df['move_quality'] == 'suboptimal').mean() * 100
                    
                    # Phase-specific performance
                    for phase in ['opening', 'middlegame', 'endgame']:
                        phase_moves = moves_df[moves_df['game_phase'] == phase]
                        if len(phase_moves) > 0:
                            model_stats[f'{phase}_cp_loss'] = phase_moves['centipawn_loss'].mean()
                            model_stats[f'{phase}_complexity'] = phase_moves['position_complexity'].mean()
                    
                    # Move bucket analysis
                    model_stats.update(self.calculate_move_bucket_stats(moves_df))
                
                stats.append(model_stats)
        
        self.aggregate_stats = pd.DataFrame(stats)
    
    def calculate_move_bucket_stats(self, moves_df):
        """Calculate performance by move number buckets"""
        bucket_stats = {}
        
        # Define move buckets
        buckets = [
            (1, 10, 'early_opening'),
            (11, 20, 'late_opening'), 
            (21, 40, 'early_middle'),
            (41, 60, 'late_middle'),
            (61, 80, 'early_end'),
            (81, 200, 'late_end')
        ]
        
        for start, end, bucket_name in buckets:
            bucket_moves = moves_df[
                (moves_df['move_number'] >= start) & 
                (moves_df['move_number'] <= end)
            ]
            
            if len(bucket_moves) > 0:
                bucket_stats[f'{bucket_name}_cp_loss'] = bucket_moves['centipawn_loss'].mean()
                bucket_stats[f'{bucket_name}_accuracy'] = (bucket_moves['move_quality'].isin(['best', 'good'])).mean() * 100
                bucket_stats[f'{bucket_name}_complexity'] = bucket_moves['position_complexity'].mean()
        
        return bucket_stats

    def setup_layout(self):
        """Setup the massively enhanced dashboard layout"""
        self.app.layout = html.Div([
            html.Div([
                html.H1("🧠 Advanced Chess GPT Analysis Dashboard", 
                       style={'textAlign': 'center', 'marginBottom': 30, 'color': '#2c3e50'}),
                
                html.Div([
                    html.Div([
                        html.Label("Select Models to Analyze:", style={'fontWeight': 'bold'}),
                        dcc.Dropdown(
                            id='model-selector',
                            options=[{'label': m.replace('_pt', '').replace('_', ' '), 
                                    'value': m} for m in self.model_list],
                            value=self.model_list[:4] if self.model_list else None,
                            multi=True
                        )
                    ], style={'width': '48%', 'display': 'inline-block'}),
                    
                    html.Div([
                        html.Label("Analysis Type:", style={'fontWeight': 'bold'}),
                        dcc.RadioItems(
                            id='analysis-type',
                            options=[
                                {'label': '📊 Overview Dashboard', 'value': 'overview'},
                                {'label': '🎯 Move Bucket Analysis', 'value': 'move_buckets'},
                                {'label': '🔥 Performance Heatmaps', 'value': 'heatmaps'},
                                {'label': '📈 Learning Patterns', 'value': 'learning'},
                                {'label': '🎲 Game Phase Deep-Dive', 'value': 'phases'},
                                {'label': '⚡ Move Quality Analysis', 'value': 'move_quality'},
                                {'label': '🧪 Advanced Comparisons', 'value': 'advanced'},
                                {'label': '🎨 Position Complexity', 'value': 'complexity'}
                            ],
                            value='overview',
                            style={'display': 'flex', 'flexDirection': 'column'}
                        )
                    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
                ], style={'marginBottom': 30}),
                
                # Control panel for granular analysis
                html.Div([
                    html.Div([
                        html.Label("Move Range Filter:", style={'fontWeight': 'bold'}),
                        dcc.RangeSlider(
                            id='move-range-slider',
                            min=1, max=100, step=1,
                            marks={i: str(i) for i in range(1, 101, 10)},
                            value=[1, 50],
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
                    
                    html.Div([
                        html.Label("Move Grouping Size (n):", style={'fontWeight': 'bold'}),
                        dcc.Slider(
                            id='move-grouping-size',
                            min=1, max=20, step=1,
                            marks={1: '1', 2: '2', 3: '3', 5: '5', 10: '10', 15: '15', 20: '20'},
                            value=1,
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '3%'}),
                    
                    html.Div([
                        html.Label("Centipawn Loss Threshold:", style={'fontWeight': 'bold'}),
                        dcc.Slider(
                            id='cp-threshold',
                            min=0, max=1000, step=10,
                            marks={i: f'{i}cp' for i in range(0, 1001, 100)},
                            value=100,
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], style={'width': '30%', 'display': 'inline-block'})
                ], style={'marginBottom': 30, 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
                
                # Main content area
                html.Div(id='main-content'),
                
                # Enhanced summary statistics
                html.Div([
                    html.H3("📈 Enhanced Model Performance Summary"),
                    html.Div(id='summary-table')
                ], style={'marginTop': 40})
            ], style={'padding': '20px'})
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks for enhanced interactivity"""
        
        @self.app.callback(
            [Output('main-content', 'children'),
             Output('summary-table', 'children')],
            [Input('model-selector', 'value'),
             Input('analysis-type', 'value'),
             Input('move-range-slider', 'value'),
             Input('move-grouping-size', 'value'),
             Input('cp-threshold', 'value')]
        )
        def update_dashboard(selected_models, analysis_type, move_range, move_grouping_size, cp_threshold):
            if not selected_models:
                return html.Div("Please select at least one model."), html.Div([])
            
            # Filter models to only those with data
            models_with_data = [m for m in selected_models if m in self.aggregate_stats['model'].values]
            if not models_with_data:
                return html.Div("No valid data available for the selected models."), html.Div([])

            # Generate main content based on analysis type
            if analysis_type == 'overview':
                main_content = self.create_overview_dashboard(models_with_data)
            elif analysis_type == 'move_buckets':
                main_content = self.create_move_bucket_analysis(models_with_data, move_range, move_grouping_size, cp_threshold)
            elif analysis_type == 'heatmaps':
                main_content = self.create_performance_heatmaps(models_with_data)
            elif analysis_type == 'learning':
                main_content = self.create_learning_patterns(models_with_data)
            elif analysis_type == 'phases':
                main_content = self.create_game_phase_analysis(models_with_data)
            elif analysis_type == 'move_quality':
                main_content = self.create_move_quality_analysis(models_with_data, move_range)
            elif analysis_type == 'advanced':
                main_content = self.create_advanced_comparisons(models_with_data)
            elif analysis_type == 'complexity':
                main_content = self.create_position_complexity_analysis(models_with_data)
            else:
                main_content = self.create_overview_dashboard(models_with_data)
            
            # Create enhanced summary table
            summary_table = self.create_enhanced_summary_table(models_with_data)
            
            return main_content, summary_table

    def create_move_bucket_analysis(self, models, move_range, move_grouping_size, cp_threshold):
        """Create granular move bucket analysis with dynamic grouping - THE CORE FEATURE YOU REQUESTED"""
        plots = []
        
        # 1. Move Bucket Performance Heatmap
        bucket_columns = [col for col in self.aggregate_stats.columns if '_cp_loss' in col and any(bucket in col for bucket in ['early_opening', 'late_opening', 'early_middle', 'late_middle', 'early_end', 'late_end'])]
        
        if bucket_columns:
            heatmap_data = []
            model_names = []
            
            for model in models:
                if model in self.aggregate_stats['model'].values:
                    model_stats = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
                    row_data = []
                    
                    for col in bucket_columns:
                        value = model_stats.get(col, np.nan)
                        row_data.append(value if not pd.isna(value) else 0)
                    
                    heatmap_data.append(row_data)
                    model_names.append(model.replace('_pt', '').replace('_', ' '))
            
            if heatmap_data:
                fig1 = go.Figure(data=go.Heatmap(
                    z=heatmap_data,
                    x=[col.replace('_cp_loss', '').replace('_', ' ').title() for col in bucket_columns],
                    y=model_names,
                    colorscale='RdYlGn_r',
                    text=[[f'{val:.1f}' for val in row] for row in heatmap_data],
                    texttemplate='%{text}',
                    textfont={"size": 12},
                    hovertemplate='Model: %{y}<br>Move Bucket: %{x}<br>Avg CP Loss: %{z:.2f}<extra></extra>'
                ))
                
                fig1.update_layout(
                    title='🎯 Move Bucket Performance Heatmap (Centipawn Loss)',
                    height=600,
                    xaxis_title='Move Buckets',
                    yaxis_title='Models'
                )
                
                plots.append(dcc.Graph(figure=fig1))
        
        # 2. Dynamic Move Grouping Performance Trends - YOUR REQUESTED FEATURE!
        if any(model in self.moves_data for model in models):
            fig2 = go.Figure()
            
            for model in models:
                if model in self.moves_data:
                    moves_df = self.moves_data[model]
                    
                    # Filter by move range
                    filtered_moves = moves_df[
                        (moves_df['move_number'] >= move_range[0]) & 
                        (moves_df['move_number'] <= move_range[1])
                    ]
                    
                    if len(filtered_moves) > 0:
                        # Create dynamic move groups based on grouping size
                        def get_move_group(move_num):
                            return ((move_num - 1) // move_grouping_size) * move_grouping_size + 1
                        
                        filtered_moves['move_group'] = filtered_moves['move_number'].apply(get_move_group)
                        
                        # Calculate average centipawn loss for each group
                        group_performance = filtered_moves.groupby('move_group')['centipawn_loss'].mean().reset_index()
                        
                        # Create group labels
                        group_labels = []
                        for group_start in group_performance['move_group']:
                            group_end = group_start + move_grouping_size - 1
                            if move_grouping_size == 1:
                                group_labels.append(f"{group_start}")
                            else:
                                group_labels.append(f"{group_start}-{group_end}")
                        
                        fig2.add_trace(go.Scatter(
                            x=group_performance['move_group'],
                            y=group_performance['centipawn_loss'],
                            mode='lines+markers',
                            name=model.replace('_pt', '').replace('_', ' '),
                            line=dict(width=3),
                            marker=dict(size=8),
                            text=group_labels,
                            hovertemplate='Move Group: %{text}<br>Avg CP Loss: %{y:.2f}<extra></extra>'
                        ))
            
            # Create custom title based on grouping size
            if move_grouping_size == 1:
                title = f'📈 Move-by-Move Performance Trends (Moves {move_range[0]}-{move_range[1]})'
                xaxis_title = 'Move Number'
            else:
                title = f'📈 Move Group Performance Trends (n={move_grouping_size}, Moves {move_range[0]}-{move_range[1]})'
                xaxis_title = f'Move Group Start (Group Size: {move_grouping_size})'
            
            fig2.update_layout(
                title=title,
                xaxis_title=xaxis_title,
                yaxis_title='Average Centipawn Loss',
                height=500,
                hovermode='x unified'
            )
            
            plots.append(dcc.Graph(figure=fig2))
        
        # 3. Group Size Impact Analysis - Shows how grouping affects variance
        if any(model in self.moves_data for model in models) and move_grouping_size > 1:
            fig_variance = go.Figure()
            
            for model in models:
                if model in self.moves_data:
                    moves_df = self.moves_data[model]
                    
                    # Filter by move range
                    filtered_moves = moves_df[
                        (moves_df['move_number'] >= move_range[0]) & 
                        (moves_df['move_number'] <= move_range[1])
                    ]
                    
                    if len(filtered_moves) > 0:
                        # Calculate both individual moves and grouped moves
                        individual_performance = filtered_moves.groupby('move_number')['centipawn_loss'].mean()
                        
                        # Create move groups
                        def get_move_group(move_num):
                            return ((move_num - 1) // move_grouping_size) * move_grouping_size + 1
                        
                        filtered_moves['move_group'] = filtered_moves['move_number'].apply(get_move_group)
                        grouped_performance = filtered_moves.groupby('move_group')['centipawn_loss'].mean()
                        
                        # Plot individual moves (more volatile)
                        fig_variance.add_trace(go.Scatter(
                            x=individual_performance.index,
                            y=individual_performance.values,
                            mode='lines',
                            name=f"{model.replace('_pt', '').replace('_', ' ')} (n=1)",
                            line=dict(width=1, dash='dot'),
                            opacity=0.6,
                            hovertemplate='Move: %{x}<br>CP Loss: %{y:.2f}<extra></extra>'
                        ))
                        
                        # Plot grouped moves (smoother)
                        fig_variance.add_trace(go.Scatter(
                            x=grouped_performance.index,
                            y=grouped_performance.values,
                            mode='lines+markers',
                            name=f"{model.replace('_pt', '').replace('_', ' ')} (n={move_grouping_size})",
                            line=dict(width=3),
                            marker=dict(size=8),
                            hovertemplate='Move Group: %{x}<br>CP Loss: %{y:.2f}<extra></extra>'
                        ))
            
            fig_variance.update_layout(
                title=f'🔍 Grouping Impact: Individual (n=1) vs Grouped (n={move_grouping_size}) Performance',
                xaxis_title='Move Number / Group Start',
                yaxis_title='Average Centipawn Loss',
                height=500,
                hovermode='x unified'
            )
            
            plots.append(dcc.Graph(figure=fig_variance))
        
        # 4. Centipawn Loss Distribution by Move Buckets
        if any(model in self.moves_data for model in models):
            fig3 = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Early Opening (1-10)', 'Late Opening (11-20)', 'Early Middle (21-40)',
                               'Late Middle (41-60)', 'Early End (61-80)', 'Late End (81+)'),
                vertical_spacing=0.12
            )
            
            buckets = [(1, 10), (11, 20), (21, 40), (41, 60), (61, 80), (81, 200)]
            positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
            
            for i, ((start, end), (row, col)) in enumerate(zip(buckets, positions)):
                for model in models:
                    if model in self.moves_data:
                        moves_df = self.moves_data[model]
                        bucket_moves = moves_df[
                            (moves_df['move_number'] >= start) & 
                            (moves_df['move_number'] <= end) &
                            (moves_df['centipawn_loss'] <= cp_threshold)
                        ]
                        
                        if len(bucket_moves) > 0:
                            fig3.add_trace(
                                go.Histogram(
                                    x=bucket_moves['centipawn_loss'],
                                    name=model.replace('_pt', '').replace('_', ' '),
                                    opacity=0.7,
                                    nbinsx=20,
                                    showlegend=(i == 0)
                                ),
                                row=row, col=col
                            )
            
            fig3.update_layout(
                title=f'📊 Centipawn Loss Distribution by Move Buckets (≤{cp_threshold}cp)',
                height=800,
                barmode='overlay'
            )
            
            plots.append(dcc.Graph(figure=fig3))
        
        return html.Div(plots)

    def create_performance_heatmaps(self, models):
        """Create comprehensive performance heatmaps"""
        plots = []
        
        # 1. Multi-Metric Performance Heatmap
        metrics = ['avg_cp_loss', 'avg_blunders', 'avg_mistakes', 'opening_acc', 'middlegame_acc', 'endgame_acc']
        
        heatmap_data = []
        for model in models:
            if model in self.aggregate_stats['model'].values:
                model_stats = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
                heatmap_data.append([
                    model_stats['avg_cp_loss'],
                    model_stats['avg_blunders'], 
                    model_stats['avg_mistakes'],
                    model_stats['opening_acc'],
                    model_stats['middlegame_acc'],
                    model_stats['endgame_acc']
                ])
        
        if heatmap_data:
            fig1 = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=['Avg CP Loss', 'Avg Blunders', 'Avg Mistakes', 'Opening Acc', 'Middlegame Acc', 'Endgame Acc'],
                y=[m.replace('_pt', '').replace('_', ' ') for m in models],
                colorscale='RdYlGn_r',
                text=[[f'{val:.1f}' for val in row] for row in heatmap_data],
                texttemplate='%{text}',
                textfont={"size": 12},
                hovertemplate='Model: %{y}<br>Metric: %{x}<br>Value: %{z:.2f}<extra></extra>'
            ))
            
            fig1.update_layout(
                title='🔥 Multi-Metric Performance Heatmap',
                xaxis_title='Performance Metrics',
                yaxis_title='Models',
                height=600
            )
            
            plots.append(dcc.Graph(figure=fig1))
        
        # 2. Move Quality Distribution Heatmap
        if any(model in self.moves_data for model in models):
            quality_data = []
            quality_metrics = ['best_move_rate', 'good_move_rate', 'suboptimal_rate']
            
            for model in models:
                if model in self.aggregate_stats['model'].values:
                    model_stats = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
                    row_data = []
                    
                    for metric in quality_metrics:
                        value = model_stats.get(metric, 0)
                        row_data.append(value)
                    
                    quality_data.append(row_data)
            
            if quality_data:
                fig2 = go.Figure(data=go.Heatmap(
                    z=quality_data,
                    x=['Best Moves (%)', 'Good Moves (%)', 'Suboptimal (%)'],
                    y=[m.replace('_pt', '').replace('_', ' ') for m in models],
                    colorscale='RdYlGn',
                    text=[[f'{val:.1f}%' for val in row] for row in quality_data],
                    texttemplate='%{text}',
                    textfont={"size": 12},
                    hovertemplate='Model: %{y}<br>Quality: %{x}<br>Rate: %{z:.2f}%<extra></extra>'
                ))
                
                fig2.update_layout(
                    title='🎯 Move Quality Distribution Heatmap',
                    height=600
                )
                
                plots.append(dcc.Graph(figure=fig2))
        
        return html.Div(plots)

    def create_learning_patterns(self, models):
        """Analyze learning patterns and trends"""
        plots = []
        
        # 1. Performance Evolution Over Games
        fig1 = go.Figure()
        
        for model in models:
            if model in self.summary_data:
                df = self.summary_data[model]
                
                # Calculate rolling average of performance
                df_sorted = df.sort_values('game_id')
                window_size = max(50, len(df) // 20)
                
                rolling_cp = df_sorted['average_centipawn_loss'].rolling(window=window_size).mean()
                
                fig1.add_trace(go.Scatter(
                    x=list(range(len(rolling_cp))),
                    y=rolling_cp,
                    mode='lines',
                    name=model.replace('_pt', '').replace('_', ' '),
                    line=dict(width=3),
                    hovertemplate='Game: %{x}<br>Rolling Avg CP Loss: %{y:.2f}<extra></extra>'
                ))
        
        fig1.update_layout(
            title='📈 Learning Patterns: Performance Evolution Over Games',
            xaxis_title='Game Number',
            yaxis_title='Rolling Average Centipawn Loss',
            height=500
        )
        
        plots.append(dcc.Graph(figure=fig1))
        
        # 2. Consistency Analysis
        fig2 = go.Figure()
        
        consistency_data = []
        for model in models:
            if model in self.summary_data:
                df = self.summary_data[model]
                
                # Calculate performance consistency (coefficient of variation)
                mean_cp = df['average_centipawn_loss'].mean()
                std_cp = df['average_centipawn_loss'].std()
                cv = (std_cp / mean_cp) * 100 if mean_cp > 0 else 0
                
                consistency_data.append({
                    'model': model.replace('_pt', '').replace('_', ' '),
                    'consistency': 100 - cv,  # Higher is more consistent
                    'mean_performance': mean_cp,
                    'std_performance': std_cp
                })
        
        if consistency_data:
            fig2.add_trace(go.Scatter(
                x=[d['mean_performance'] for d in consistency_data],
                y=[d['consistency'] for d in consistency_data],
                mode='markers+text',
                text=[d['model'] for d in consistency_data],
                textposition='top center',
                marker=dict(size=15, color='lightblue', line=dict(width=2)),
                hovertemplate='Model: %{text}<br>Avg Performance: %{x:.2f}<br>Consistency: %{y:.2f}%<extra></extra>'
            ))
            
            fig2.update_layout(
                title='🎯 Performance vs Consistency Analysis',
                xaxis_title='Average Centipawn Loss',
                yaxis_title='Consistency Score (%)',
                height=500
            )
            
            plots.append(dcc.Graph(figure=fig2))
        
        return html.Div(plots)

    def create_game_phase_analysis(self, models):
        """Deep dive into game phase performance"""
        plots = []
        
        # 1. Phase Performance Comparison
        phases = ['opening', 'middlegame', 'endgame']
        
        fig1 = go.Figure()
        
        for model in models:
            if model in self.aggregate_stats['model'].values:
                model_stats = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
                
                phase_losses = []
                for phase in phases:
                    cp_loss = model_stats.get(f'{phase}_cp_loss', 0)
                    phase_losses.append(cp_loss)
                
                fig1.add_trace(go.Scatter(
                    x=phases,
                    y=phase_losses,
                    mode='lines+markers',
                    name=model.replace('_pt', '').replace('_', ' '),
                    line=dict(width=3),
                    marker=dict(size=10),
                    hovertemplate='Phase: %{x}<br>Avg CP Loss: %{y:.2f}<extra></extra>'
                ))
        
        fig1.update_layout(
            title='🎲 Game Phase Performance Comparison',
            xaxis_title='Game Phase',
            yaxis_title='Average Centipawn Loss',
            height=500
        )
        
        plots.append(dcc.Graph(figure=fig1))
        
        # 2. Phase Complexity vs Performance
        if any(model in self.moves_data for model in models):
            fig2 = make_subplots(
                rows=1, cols=3,
                subplot_titles=('Opening', 'Middlegame', 'Endgame')
            )
            
            for i, phase in enumerate(phases):
                for model in models:
                    if model in self.moves_data:
                        moves_df = self.moves_data[model]
                        phase_moves = moves_df[moves_df['game_phase'] == phase]
                        
                        if len(phase_moves) > 0:
                            fig2.add_trace(
                                go.Scatter(
                                    x=phase_moves['position_complexity'],
                                    y=phase_moves['centipawn_loss'],
                                    mode='markers',
                                    name=model.replace('_pt', '').replace('_', ' '),
                                    opacity=0.6,
                                    showlegend=(i == 0),
                                    hovertemplate='Complexity: %{x:.2f}<br>CP Loss: %{y:.2f}<extra></extra>'
                                ),
                                row=1, col=i+1
                            )
            
            fig2.update_layout(
                title='🧠 Position Complexity vs Performance by Phase',
                height=600
            )
            
            plots.append(dcc.Graph(figure=fig2))
        
        return html.Div(plots)

    def create_move_quality_analysis(self, models, move_range):
        """Analyze move quality patterns"""
        plots = []
        
        if any(model in self.moves_data for model in models):
            # 1. Move Quality Distribution
            fig1 = go.Figure()
            
            quality_categories = ['best', 'good', 'suboptimal', 'mistake', 'blunder']
            
            for model in models:
                if model in self.moves_data:
                    moves_df = self.moves_data[model]
                    
                    # Filter by move range
                    filtered_moves = moves_df[
                        (moves_df['move_number'] >= move_range[0]) & 
                        (moves_df['move_number'] <= move_range[1])
                    ]
                    
                    if len(filtered_moves) > 0:
                        quality_counts = filtered_moves['move_quality'].value_counts()
                        quality_percentages = []
                        
                        for quality in quality_categories:
                            count = quality_counts.get(quality, 0)
                            percentage = (count / len(filtered_moves)) * 100
                            quality_percentages.append(percentage)
                        
                        fig1.add_trace(go.Bar(
                            x=quality_categories,
                            y=quality_percentages,
                            name=model.replace('_pt', '').replace('_', ' '),
                            hovertemplate='Quality: %{x}<br>Percentage: %{y:.2f}%<extra></extra>'
                        ))
            
            fig1.update_layout(
                title=f'⚡ Move Quality Distribution (Moves {move_range[0]}-{move_range[1]})',
                xaxis_title='Move Quality',
                yaxis_title='Percentage of Moves',
                height=500,
                barmode='group'
            )
            
            plots.append(dcc.Graph(figure=fig1))
            
            # 2. Move Quality Score vs Centipawn Loss
            fig2 = go.Figure()
            
            for model in models:
                if model in self.moves_data:
                    moves_df = self.moves_data[model]
                    
                    # Sample data for performance
                    sample_moves = moves_df.sample(min(5000, len(moves_df)))
                    
                    fig2.add_trace(go.Scatter(
                        x=sample_moves['move_quality_score'],
                        y=sample_moves['centipawn_loss'],
                        mode='markers',
                        name=model.replace('_pt', '').replace('_', ' '),
                        opacity=0.6,
                        hovertemplate='Quality Score: %{x}<br>CP Loss: %{y:.2f}<extra></extra>'
                    ))
            
            fig2.update_layout(
                title='🎯 Move Quality Score vs Centipawn Loss',
                xaxis_title='Move Quality Score (0-100)',
                yaxis_title='Centipawn Loss',
                height=500
            )
            
            plots.append(dcc.Graph(figure=fig2))
        
        return html.Div(plots)

    def create_advanced_comparisons(self, models):
        """Advanced statistical comparisons"""
        plots = []
        
        # 1. Statistical Significance Testing
        if len(models) >= 2:
            significance_results = []
            
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    model1, model2 = models[i], models[j]
                    
                    if model1 in self.summary_data and model2 in self.summary_data:
                        data1 = self.summary_data[model1]['average_centipawn_loss'].values
                        data2 = self.summary_data[model2]['average_centipawn_loss'].values
                        
                        # Perform t-test
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        effect_size = (np.mean(data1) - np.mean(data2)) / np.sqrt((np.var(data1) + np.var(data2)) / 2)
                        
                        significance_results.append({
                            'Model 1': model1.replace('_pt', '').replace('_', ' '),
                            'Model 2': model2.replace('_pt', '').replace('_', ' '),
                            'Mean Diff': np.mean(data1) - np.mean(data2),
                            'T-statistic': t_stat,
                            'P-value': p_value,
                            'Effect Size': effect_size,
                            'Significant': 'Yes' if p_value < 0.05 else 'No',
                            'Magnitude': 'Large' if abs(effect_size) > 0.8 else 'Medium' if abs(effect_size) > 0.5 else 'Small'
                        })
            
            if significance_results:
                significance_table = dash_table.DataTable(
                    data=significance_results,
                    columns=[{"name": i, "id": i, "type": "numeric", "format": {"specifier": ".3f"}} if i in ['Mean Diff', 'T-statistic', 'P-value', 'Effect Size'] else {"name": i, "id": i} for i in significance_results[0].keys()],
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{Significant} = Yes'},
                            'backgroundColor': '#d4edda',
                            'color': 'black',
                        },
                        {
                            'if': {'filter_query': '{Magnitude} = Large'},
                            'backgroundColor': '#f8d7da',
                            'color': 'black',
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                )
                
                plots.append(html.Div([
                    html.H4("🧪 Statistical Significance Testing"),
                    html.P("T-tests comparing centipawn loss between models. Green = statistically significant (p < 0.05), Red = large effect size."),
                    significance_table
                ], style={'margin': '20px'}))
        
        return html.Div(plots)

    def create_position_complexity_analysis(self, models):
        """Analyze performance vs position complexity"""
        plots = []
        
        if any(model in self.moves_data for model in models):
            # 1. Complexity vs Performance Scatter
            fig1 = go.Figure()
            
            for model in models:
                if model in self.moves_data:
                    moves_df = self.moves_data[model]
                    
                    # Sample data for performance
                    sample_moves = moves_df.sample(min(3000, len(moves_df)))
                    
                    fig1.add_trace(go.Scatter(
                        x=sample_moves['position_complexity'],
                        y=sample_moves['centipawn_loss'],
                        mode='markers',
                        name=model.replace('_pt', '').replace('_', ' '),
                        opacity=0.6,
                        hovertemplate='Complexity: %{x:.2f}<br>CP Loss: %{y:.2f}<extra></extra>'
                    ))
            
            fig1.update_layout(
                title='🎨 Position Complexity vs Performance',
                xaxis_title='Position Complexity',
                yaxis_title='Centipawn Loss',
                height=500
            )
            
            plots.append(dcc.Graph(figure=fig1))
            
            # 2. Complexity Distribution by Model
            fig2 = go.Figure()
            
            for model in models:
                if model in self.moves_data:
                    moves_df = self.moves_data[model]
                    
                    fig2.add_trace(go.Histogram(
                        x=moves_df['position_complexity'],
                        name=model.replace('_pt', '').replace('_', ' '),
                        opacity=0.7,
                        nbinsx=30
                    ))
            
            fig2.update_layout(
                title='📊 Position Complexity Distribution',
                xaxis_title='Position Complexity',
                yaxis_title='Frequency',
                height=500,
                barmode='overlay'
            )
            
            plots.append(dcc.Graph(figure=fig2))
        
        return html.Div(plots)

    def create_overview_dashboard(self, models):
        """Create enhanced overview dashboard"""
        plots = []
        
        # 1. Performance Overview
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Centipawn Loss', 'Move Quality Distribution',
                          'Game Phase Performance', 'Consistency Analysis')
        )
        
        # Centipawn loss comparison
        for i, model in enumerate(models):
            if model in self.summary_data:
                df = self.summary_data[model]
                fig1.add_trace(
                    go.Box(y=df['average_centipawn_loss'], name=model.replace('_pt', '').replace('_', ' '),
                          marker_color=px.colors.qualitative.Set3[i % 10]),
                    row=1, col=1
                )
        
        # Move quality if available
        if any(model in self.moves_data for model in models):
            for i, model in enumerate(models):
                if model in self.moves_data and model in self.aggregate_stats['model'].values:
                    model_stats = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
                    
                    fig1.add_trace(
                        go.Bar(
                            x=['Best', 'Good', 'Suboptimal'],
                            y=[model_stats.get('best_move_rate', 0), 
                               model_stats.get('good_move_rate', 0),
                               model_stats.get('suboptimal_rate', 0)],
                            name=model.replace('_pt', '').replace('_', ' '),
                            marker_color=px.colors.qualitative.Set3[i % 10]
                        ),
                        row=1, col=2
                    )
        
        fig1.update_layout(height=800, title_text="📊 Performance Overview Dashboard")
        plots.append(dcc.Graph(figure=fig1))
        
        return html.Div(plots)

    def create_enhanced_summary_table(self, models):
        """Create enhanced summary statistics table"""
        table_data = []
        
        for model in models:
            if model in self.aggregate_stats['model'].values:
                stats = self.aggregate_stats[self.aggregate_stats['model'] == model].iloc[0]
                
                row_data = {
                    'Model': model.replace('_pt', '').replace('_', ' '),
                    'Games': f"{stats['games']:,}",
                    'Avg CP Loss': f"{stats['avg_cp_loss']:.1f}",
                    'Blunders/Game': f"{stats['avg_blunders']:.2f}",
                    'Opening Acc': f"{stats['opening_acc']:.1f}%",
                    'Middlegame Acc': f"{stats['middlegame_acc']:.1f}%",
                    'Endgame Acc': f"{stats['endgame_acc']:.1f}%"
                }
                
                # Add move quality stats if available
                if 'best_move_rate' in stats:
                    row_data['Best Moves'] = f"{stats['best_move_rate']:.1f}%"
                    row_data['Good Moves'] = f"{stats['good_move_rate']:.1f}%"
                
                table_data.append(row_data)
        
        if table_data:
            return dash_table.DataTable(
                data=table_data,
                columns=[{"name": i, "id": i} for i in table_data[0].keys()],
                style_cell={'textAlign': 'center'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            )
        
        return html.Div("No data available for selected models")
    
    def run(self, debug=False, port=8050):
        """Run the enhanced dashboard"""
        print("🚀 Starting Advanced Chess GPT Analysis Dashboard...")
        print(f"📊 Loaded {len(self.model_list)} models with move-level data")
        print(f"🌐 Dashboard will be available at: http://localhost:{port}")
        print("🎯 Features: Move buckets, Performance heatmaps, Learning patterns, Deep analysis")
        self.app.run(debug=debug, port=port)

if __name__ == "__main__":
    dashboard = AdvancedChessAnalysisDashboard()
    dashboard.run() 