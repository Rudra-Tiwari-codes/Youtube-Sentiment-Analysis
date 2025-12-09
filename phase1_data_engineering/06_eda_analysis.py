"""
Exploratory Data Analysis (EDA)
Generate visualizations for Phase 1 data engineering report.

1. Language distribution (pie chart)
2. Sentiment distribution (bar chart)
3. Comment length vs sentiment (scatter plot with trend lines)
4. Temporal trends (comments over time)
5. Language × Sentiment cross-tabulation (heatmap)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import logging
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def create_language_distribution_chart(df: pd.DataFrame, output_path: str):
    """Create interactive pie chart of language distribution"""
    logger.info("Creating language distribution chart...")
    
    lang_counts = df['language'].value_counts()
    
    fig = px.pie(
        values=lang_counts.values,
        names=lang_counts.index,
        title='Language Distribution in YouTube Comments<br><sub>Indian Unemployment Discourse Dataset (131,608 comments)</sub>',
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.3  # Donut chart
    )
    
    fig.update_traces(
        textposition='auto',
        textinfo='label+percent+value',
        textfont_size=14
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        font=dict(size=14)
    )
    
    fig.write_html(output_path)
    logger.info(f"Saved: {output_path}")


def create_sentiment_distribution_chart(df: pd.DataFrame, output_path: str):
    """Create bar chart of sentiment distribution"""
    logger.info("Creating sentiment distribution chart...")
    
    sent_counts = df['sentiment'].value_counts()
    colors = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
    
    fig = go.Figure(data=[
        go.Bar(
            x=sent_counts.index,
            y=sent_counts.values,
            text=sent_counts.values,
            texttemplate='%{text:,}<br>(%{y:.1%})',
            textposition='auto',
            marker=dict(
                color=[colors.get(x, '#3498db') for x in sent_counts.index],
                line=dict(color='white', width=2)
            )
        )
    ])
    
    fig.update_layout(
        title='Sentiment Distribution in Dataset<br><sub>VADER-based labels for 131,608 YouTube comments</sub>',
        xaxis_title='Sentiment Category',
        yaxis_title='Number of Comments',
        height=500,
        font=dict(size=14),
        showlegend=False
    )
    
    fig.update_yaxes(tickformat=',')
    
    fig.write_html(output_path)
    logger.info(f"Saved: {output_path}")


def create_length_vs_sentiment_chart(df: pd.DataFrame, output_path: str):
    """Create scatter plot showing correlation between comment length and sentiment"""
    logger.info("Creating length vs sentiment correlation chart:  ")
    
    # Sample for visualization (plotting 131k points is slow)
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    # Create color mapping
    color_map = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
    
    fig = px.scatter(
        df_sample,
        x='cleaned_length',
        y='vader_compound',
        color='sentiment',
        color_discrete_map=color_map,
        title=f'Comment Length vs Sentiment Polarity<br><sub>Random sample of {sample_size:,} comments</sub>',
        labels={
            'cleaned_length': 'Comment Length (characters)',
            'vader_compound': 'VADER Compound Score',
            'sentiment': 'Sentiment'
        },
        opacity=0.6,
        trendline='lowess'  # Local regression trend line
    )
    
    # Add horizontal lines for sentiment thresholds
    fig.add_hline(y=0.05, line_dash="dash", line_color="green", 
                  annotation_text="Positive threshold", annotation_position="right")
    fig.add_hline(y=-0.05, line_dash="dash", line_color="red",
                  annotation_text="Negative threshold", annotation_position="right")
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        height=600,
        font=dict(size=14),
        hovermode='closest'
    )
    
    fig.update_traces(marker=dict(size=5))
    
    fig.write_html(output_path)
    logger.info(f"Saved: {output_path}")
    
    # Calculate correlation
    correlation = df['cleaned_length'].corr(df['vader_compound'])
    logger.info(f"Length-Sentiment correlation: {correlation:.4f}")


def create_temporal_trends_chart(df: pd.DataFrame, output_path: str):
    """Create line chart showing comments over time"""
    logger.info("Creating temporal trends chart.")
    
    # Convert created_date to datetime
    df['date'] = pd.to_datetime(df['created_date'])
    df['year_month'] = df['date'].dt.to_period('M').astype(str)
    
    # Count comments by month
    temporal_data = df.groupby('year_month').size().reset_index(name='count')
    
    # Also get sentiment breakdown by month
    sentiment_temporal = df.groupby(['year_month', 'sentiment']).size().unstack(fill_value=0)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Total Comments Over Time',
            'Sentiment Distribution Over Time'
        ),
        row_heights=[0.4, 0.6],
        vertical_spacing=0.15
    )
    
    # Total comments line
    fig.add_trace(
        go.Scatter(
            x=temporal_data['year_month'],
            y=temporal_data['count'],
            mode='lines+markers',
            name='Total Comments',
            line=dict(color='#3498db', width=2),
            fill='tozeroy'
        ),
        row=1, col=1
    )
    
    # Sentiment breakdown
    colors = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
    for sentiment in ['Positive', 'Neutral', 'Negative']:
        if sentiment in sentiment_temporal.columns:
            fig.add_trace(
                go.Scatter(
                    x=sentiment_temporal.index,
                    y=sentiment_temporal[sentiment],
                    mode='lines',
                    name=sentiment,
                    line=dict(color=colors[sentiment], width=2),
                    stackgroup='one'
                ),
                row=2, col=1
            )
    
    fig.update_xaxes(title_text="Month", row=2, col=1)
    fig.update_yaxes(title_text="Number of Comments", row=1, col=1)
    fig.update_yaxes(title_text="Number of Comments", row=2, col=1)
    
    fig.update_layout(
        title_text='Temporal Analysis of YouTube Comments<br><sub>Indian Unemployment Discourse (2019-2025)</sub>',
        height=900,
        font=dict(size=12),
        showlegend=True,
        hovermode='x unified'
    )
    
    # Rotate x-axis labels for better readability
    fig.update_xaxes(tickangle=45)
    
    fig.write_html(output_path)
    logger.info(f"Saved: {output_path}")


def create_language_sentiment_heatmap(df: pd.DataFrame, output_path: str):
    """Create heatmap showing language × sentiment cross-tabulation"""
    logger.info("Creating language × sentiment heatmap...")
    
    # Create cross-tabulation
    crosstab = pd.crosstab(df['language'], df['sentiment'])
    
    # Calculate percentages
    crosstab_pct = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=crosstab_pct.values,
        x=crosstab_pct.columns,
        y=crosstab_pct.index,
        text=crosstab.values,
        texttemplate='%{text:,}<br>(%{z:.1f}%)',
        textfont={"size": 14},
        colorscale='Blues',
        colorbar=dict(title="Percentage")
    ))
    
    fig.update_layout(
        title='Sentiment Distribution by Language<br><sub>Cross-tabulation of 131,608 comments</sub>',
        xaxis_title='Sentiment',
        yaxis_title='Language',
        height=500,
        font=dict(size=14)
    )
    
    fig.write_html(output_path)
    logger.info(f"Saved: {output_path}")


# Needed for research summary/description
def generate_summary_statistics(df: pd.DataFrame) -> dict:
    """Generate comprehensive summary statistics"""
    logger.info("Calculating summary statistics...")
    
    stats = {
        'total_records': len(df),
        'unique_authors': df['author'].nunique(),
        'date_range': {
            'start': df['created_date'].min(),
            'end': df['created_date'].max()
        },
        'language_distribution': df['language'].value_counts().to_dict(),
        'sentiment_distribution': df['sentiment'].value_counts().to_dict(),
        'avg_comment_length': df['cleaned_length'].mean(),
        'median_comment_length': df['cleaned_length'].median(),
        'avg_vader_compound': df['vader_compound'].mean(),
        'sentiment_by_language': df.groupby('language')['sentiment'].value_counts().to_dict()
    }
    
    return stats


def main():

    INPUT_PATH = "./data/processed/comments_labeled.csv"
    OUTPUT_DIR = "./phase1_data_engineering/figures"
    
    logger.info("="*80)
    logger.info("EDA STARTED")
    logger.info("="*80)
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    logger.info(f"Loaded {len(df):,} records")
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    create_language_distribution_chart(
        df, 
        f"{OUTPUT_DIR}/language_distribution.html"
    )
    
    create_sentiment_distribution_chart(
        df,
        f"{OUTPUT_DIR}/sentiment_distribution.html"
    )
    
    create_length_vs_sentiment_chart(
        df,
        f"{OUTPUT_DIR}/length_vs_sentiment.html"
    )
    
    create_temporal_trends_chart(
        df,
        f"{OUTPUT_DIR}/temporal_trends.html"
    )
    
    create_language_sentiment_heatmap(
        df,
        f"{OUTPUT_DIR}/language_sentiment_heatmap.html"
    )
    
    # Generate summary statistics
    stats = generate_summary_statistics(df)
    
    logger.info("="*80)
    logger.info("EDA COMPLETE ")
    logger.info("="*80)
    
    # Print summary
    '''
    print("\n" + "="*80)
    print(" EXPLORATORY DATA ANALYSIS SUMMARY")
    print("="*80)
    print(f" Total records analyzed: {stats['total_records']:,}")
    print(f" Unique authors: {stats['unique_authors']:,}")
    print(f" Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    
    print(f"\n Key Statistics:")
    print(f"   Average comment length: {stats['avg_comment_length']:.1f} characters")
    print(f"   Median comment length: {stats['median_comment_length']:.0f} characters")
    print(f"   Average VADER score: {stats['avg_vader_compound']:.4f}")
    
    print(f"\n Language Distribution:")
    for lang, count in stats['language_distribution'].items():
        pct = (count / stats['total_records']) * 100
        print(f"   {lang:12}: {count:6,} ({pct:5.2f}%)")
    
    print(f"\n Sentiment Distribution:")
    for sent, count in stats['sentiment_distribution'].items():
        pct = (count / stats['total_records']) * 100
        print(f"   {sent:10}: {count:6,} ({pct:5.2f}%)")
    
    print(f"\n Visualizations saved to: {OUTPUT_DIR}")
    print(f"   - language_distribution.html")
    print(f"   - sentiment_distribution.html")
    print(f"   - length_vs_sentiment.html")
    print(f"   - temporal_trends.html")
    print(f"   - language_sentiment_heatmap.html")
    print("="*80 + "\n")
'''

if __name__ == "__main__":
    main()
