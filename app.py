"""
🎓 Skill Gap Awareness Dashboard
================================
A Streamlit application for personalized learning recommendations
based on student archetypes and engagement patterns.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Skill Gap Awareness Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .archetype-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
    }
    .explanation-text {
        color: #555;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Data paths - organized subdirectories
DATA_DIR = Path("cleaned_data")
CLUSTER_DIR = DATA_DIR / "cluster_results"
RECOMMENDATION_DIR = DATA_DIR / "recommendation_data"
FEATURE_DIR = DATA_DIR / "feature_data"
MODEL_DIR = DATA_DIR / "model_artifacts"

@st.cache_data
def load_data():
    """Load all required data and models."""
    data = {}

    # Load cluster assignments
    data['cluster_assignments'] = pd.read_csv(CLUSTER_DIR / 'cluster_assignments.csv')

    # Load interaction matrix
    data['interaction_matrix'] = pd.read_csv(RECOMMENDATION_DIR / 'interaction_matrix_full.csv', index_col=0)

    # Load success templates
    data['success_templates'] = pd.read_csv(RECOMMENDATION_DIR / 'success_templates_v2.csv', index_col=0)

    # Load activity correlations
    if (RECOMMENDATION_DIR / 'activity_success_correlations.csv').exists():
        data['activity_correlations'] = pd.read_csv(RECOMMENDATION_DIR / 'activity_success_correlations.csv', index_col=0)

    # Load archetype success rates
    if (CLUSTER_DIR / 'archetype_success_rates.csv').exists():
        data['archetype_success_rates'] = pd.read_csv(CLUSTER_DIR / 'archetype_success_rates.csv', index_col=0)

    # Load course templates
    if (RECOMMENDATION_DIR / 'course_templates.json').exists():
        with open(RECOMMENDATION_DIR / 'course_templates.json', 'r') as f:
            data['course_templates'] = json.load(f)

    # Load course-archetype templates
    if (RECOMMENDATION_DIR / 'course_archetype_templates.json').exists():
        with open(RECOMMENDATION_DIR / 'course_archetype_templates.json', 'r') as f:
            data['course_archetype_templates'] = json.load(f)

    return data

def get_archetype_color(archetype):
    """Return color for each archetype."""
    colors = {
        'High Performer': '#2ecc71',
        'Talented but Inconsistent': '#f39c12',
        'Moderate Performer': '#3498db',
        'Early Struggler': '#e74c3c',
        'Disengaged At-Risk': '#9b59b6'
    }
    return colors.get(archetype, '#95a5a6')

def get_archetype_emoji(archetype):
    """Return emoji for each archetype."""
    emojis = {
        'High Performer': '🌟',
        'Talented but Inconsistent': '⚡',
        'Moderate Performer': '📚',
        'Early Struggler': '🔧',
        'Disengaged At-Risk': '⚠️'
    }
    return emojis.get(archetype, '👤')

def calculate_recommendations(student_data, template, correlations=None):
    """Calculate skill gap recommendations for a student."""
    recommendations = []

    for activity in template.index:
        if activity in student_data.index:
            current = student_data[activity]
            target = template[activity]
            gap = target - current

            if gap > 0.05:  # Significant gap
                corr = 0
                if correlations is not None and activity in correlations.index:
                    corr = correlations.loc[activity, 'correlation'] if 'correlation' in correlations.columns else 0

                recommendations.append({
                    'activity': activity,
                    'current': current,
                    'target': target,
                    'gap': gap,
                    'correlation': corr,
                    'priority_score': gap * (1 + abs(corr))
                })

    # Sort by priority
    recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
    return recommendations

def generate_explanation(rec, archetype):
    """Generate human-readable explanation for a recommendation."""
    explanations = []

    # Gap explanation
    gap_percent = (rec['gap'] / max(rec['target'], 0.01)) * 100
    explanations.append(f"📊 You're **{gap_percent:.0f}%** below the target for {rec['activity']} engagement.")

    # Correlation explanation
    if abs(rec['correlation']) > 0.1:
        strength = "strongly" if abs(rec['correlation']) > 0.3 else "moderately"
        direction = "positively" if rec['correlation'] > 0 else "negatively"
        explanations.append(f"📈 This activity is **{strength} {direction}** correlated with course success (r={rec['correlation']:.2f}).")

    # Impact estimate
    if rec['correlation'] > 0.1:
        impact = rec['gap'] * rec['correlation'] * 50
        if impact > 3:
            explanations.append(f"🎯 Improving this could increase your success probability by ~**{impact:.0f}%**.")

    return explanations

def get_action_text(activity):
    """Get specific action text for an activity."""
    actions = {
        'quiz': "Complete more practice quizzes to test your understanding.",
        'forumng': "Participate in discussion forums - ask questions and help others.",
        'oucontent': "Spend more time reading course materials and taking notes.",
        'resource': "Download and review additional learning resources.",
        'subpage': "Explore more course subpages for supplementary content.",
        'homepage': "Check the course homepage regularly for updates.",
        'url': "Visit external resources linked in the course.",
        'ouwiki': "Contribute to or read the course wiki.",
        'oucollaborate': "Join collaborative sessions with peers.",
        'ouelluminate': "Attend live online sessions.",
        'glossary': "Review the course glossary for key terms.",
        'dataplus': "Explore data visualizations and interactive content.",
        'questionnaire': "Complete course questionnaires for feedback.",
        'page': "Read additional course pages.",
        'folder': "Browse course folders for materials.",
        'externalquiz': "Take external quizzes for extra practice.",
        'htmlactivity': "Engage with interactive HTML activities.",
        'dualpane': "Use the dual-pane view for content comparison.",
        'repeatactivity': "Revisit and repeat key activities.",
        'sharedsubpage': "Collaborate on shared subpages."
    }
    return actions.get(activity, f"Increase your engagement with {activity}.")

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🎓 Skill Gap Awareness Dashboard</h1>', unsafe_allow_html=True)

    # Load data
    try:
        data = load_data()
        st.success("✅ Data loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("Please ensure all data files are in the 'cleaned_data' directory.")
        return

    # Sidebar - Student Selection
    st.sidebar.header("🔍 Student Selection")

    # Get unique archetypes and courses
    archetypes = data['cluster_assignments']['archetype'].unique().tolist()

    # Determine ID column
    id_col = 'student_course_id' if 'student_course_id' in data['cluster_assignments'].columns else 'id_student'

    # Filter by archetype
    selected_archetype = st.sidebar.selectbox(
        "Filter by Archetype",
        ["All"] + archetypes
    )

    # Filter students
    if selected_archetype != "All":
        filtered_students = data['cluster_assignments'][
            data['cluster_assignments']['archetype'] == selected_archetype
        ][id_col].tolist()
    else:
        filtered_students = data['cluster_assignments'][id_col].tolist()

    # Student selector
    selected_student = st.sidebar.selectbox(
        "Select Student",
        filtered_students[:1000],  # Limit for performance
        format_func=lambda x: f"{x}"
    )

    if not selected_student:
        st.warning("Please select a student from the sidebar.")
        return

    # Get student info
    student_info = data['cluster_assignments'][
        data['cluster_assignments'][id_col] == selected_student
    ].iloc[0]

    archetype = student_info['archetype']
    final_result = student_info.get('final_result', 'Unknown')

    # Extract course from student_course_id
    if '_' in str(selected_student):
        parts = str(selected_student).split('_')
        course = '_'.join(parts[1:]) if len(parts) > 1 else 'Unknown'
    else:
        course = 'Unknown'

    # Get student engagement data
    if str(selected_student) in data['interaction_matrix'].index:
        student_engagement = data['interaction_matrix'].loc[str(selected_student)]
    else:
        st.warning(f"No engagement data found for student {selected_student}")
        return

    # Main content area
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(f"### {get_archetype_emoji(archetype)} Student Profile")
        st.markdown(f"**Student ID:** `{selected_student}`")
        st.markdown(f"**Course:** `{course}`")

    with col2:
        st.markdown(f"### 🎯 Archetype")
        archetype_color = get_archetype_color(archetype)
        st.markdown(
            f'<div style="background-color: {archetype_color}; color: white; '
            f'padding: 1rem; border-radius: 10px; text-align: center; font-weight: bold;">'
            f'{archetype}</div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(f"### 📊 Outcome")
        result_color = '#2ecc71' if final_result in ['Pass', 'Distinction'] else '#e74c3c'
        st.markdown(
            f'<div style="background-color: {result_color}; color: white; '
            f'padding: 1rem; border-radius: 10px; text-align: center; font-weight: bold;">'
            f'{final_result}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Archetype Success Rate
    if 'archetype_success_rates' in data and archetype in data['archetype_success_rates'].index:
        success_rate = data['archetype_success_rates'].loc[archetype, 'success_rate']
        st.metric(
            "Archetype Success Rate",
            f"{success_rate*100:.1f}%",
            help=f"Historical success rate for students in the '{archetype}' archetype"
        )

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Engagement", "🎯 Recommendations", "👥 Peer Comparison", "📈 Activity Correlations"])

    with tab1:
        st.subheader("Current Engagement Levels")

        # Radar chart for engagement
        activities = student_engagement.index.tolist()
        values = student_engagement.values.tolist()

        # Filter out zero values for cleaner visualization
        non_zero_mask = [v > 0.01 for v in values]
        if any(non_zero_mask):
            filtered_activities = [a for a, m in zip(activities, non_zero_mask) if m]
            filtered_values = [v for v, m in zip(values, non_zero_mask) if m]
        else:
            filtered_activities = activities[:10]
            filtered_values = values[:10]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=filtered_values,
            theta=filtered_activities,
            fill='toself',
            name='Your Engagement',
            fillcolor='rgba(31, 119, 180, 0.3)',
            line=dict(color='#1f77b4')
        ))

        # Add success template if available
        if archetype in data['success_templates'].index:
            template = data['success_templates'].loc[archetype]
            template_values = [template.get(a, 0) for a in filtered_activities]

            fig.add_trace(go.Scatterpolar(
                r=template_values,
                theta=filtered_activities,
                fill='toself',
                name='Success Template',
                fillcolor='rgba(46, 204, 113, 0.3)',
                line=dict(color='#2ecc71')
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Engagement vs Success Template",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Bar chart for all activities
        st.subheader("Detailed Engagement by Activity")

        engagement_df = pd.DataFrame({
            'Activity': activities,
            'Engagement': values
        }).sort_values('Engagement', ascending=True)

        fig_bar = px.bar(
            engagement_df,
            x='Engagement',
            y='Activity',
            orientation='h',
            color='Engagement',
            color_continuous_scale='Blues',
            title="Engagement Level by Activity"
        )
        fig_bar.update_layout(height=600)
        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        st.subheader("🎯 Personalized Recommendations")

        # Get recommendations
        if archetype in data['success_templates'].index:
            template = data['success_templates'].loc[archetype]
            correlations = data.get('activity_correlations', None)
            recommendations = calculate_recommendations(student_engagement, template, correlations)

            if not recommendations:
                st.success("🎉 Great job! You're meeting or exceeding targets in all areas!")
            else:
                # Summary
                total_gap = sum(r['gap'] for r in recommendations)
                st.info(f"📋 **Summary:** Focus on these {min(len(recommendations), 5)} areas. Total engagement gap: {total_gap:.2f}")

                # Display top recommendations
                for i, rec in enumerate(recommendations[:5], 1):
                    with st.expander(f"**Priority {i}: {rec['activity'].upper()}** (Gap: {rec['gap']:.3f})", expanded=(i<=3)):
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            st.metric("Current", f"{rec['current']:.3f}")
                            st.metric("Target", f"{rec['target']:.3f}")

                            # Progress bar
                            progress = min(rec['current'] / max(rec['target'], 0.01), 1.0)
                            st.progress(progress)

                        with col2:
                            st.markdown("**💡 Why This Matters:**")
                            explanations = generate_explanation(rec, archetype)
                            for exp in explanations:
                                st.markdown(f"- {exp}")

                            st.markdown("**✅ Recommended Action:**")
                            st.info(get_action_text(rec['activity']))
        else:
            st.warning(f"No success template available for archetype: {archetype}")

    with tab3:
        st.subheader("👥 Peer Comparison")

        if archetype in data['success_templates'].index:
            template = data['success_templates'].loc[archetype]

            # Create comparison dataframe
            comparison_data = []
            for activity in student_engagement.index:
                if activity in template.index:
                    comparison_data.append({
                        'Activity': activity,
                        'You': student_engagement[activity],
                        'Successful Peers': template[activity],
                        'Gap': template[activity] - student_engagement[activity]
                    })

            comparison_df = pd.DataFrame(comparison_data)
            comparison_df['Status'] = comparison_df['Gap'].apply(
                lambda x: '🟢 Above Target' if x <= 0 else ('🟡 Close' if x < 0.1 else '🔴 Below Target')
            )

            # Grouped bar chart
            fig_comparison = go.Figure()

            fig_comparison.add_trace(go.Bar(
                name='Your Engagement',
                x=comparison_df['Activity'],
                y=comparison_df['You'],
                marker_color='#3498db'
            ))

            fig_comparison.add_trace(go.Bar(
                name='Successful Peers',
                x=comparison_df['Activity'],
                y=comparison_df['Successful Peers'],
                marker_color='#2ecc71'
            ))

            fig_comparison.update_layout(
                barmode='group',
                title="Your Engagement vs Successful Peers",
                xaxis_tickangle=-45,
                height=500
            )

            st.plotly_chart(fig_comparison, use_container_width=True)

            # Summary table
            st.markdown("### Detailed Comparison")
            st.dataframe(
                comparison_df[['Activity', 'You', 'Successful Peers', 'Gap', 'Status']].style.format({
                    'You': '{:.3f}',
                    'Successful Peers': '{:.3f}',
                    'Gap': '{:+.3f}'
                }),
                use_container_width=True
            )

    with tab4:
        st.subheader("📈 Activity-Success Correlations")

        if 'activity_correlations' in data:
            corr_df = data['activity_correlations'].reset_index()
            corr_df.columns = ['Activity', 'Correlation']
            corr_df = corr_df.sort_values('Correlation', ascending=True)

            # Color based on correlation strength
            colors = ['#2ecc71' if c > 0.3 else ('#3498db' if c > 0.1 else '#95a5a6') 
                      for c in corr_df['Correlation']]

            fig_corr = go.Figure(go.Bar(
                x=corr_df['Correlation'],
                y=corr_df['Activity'],
                orientation='h',
                marker_color=colors
            ))

            fig_corr.update_layout(
                title="Which Activities Are Most Correlated with Success?",
                xaxis_title="Correlation with Course Success",
                height=600
            )

            st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("""
            **How to interpret:**
            - 🟢 **Strong correlation (>0.3)**: Activities strongly linked to success
            - 🔵 **Moderate correlation (0.1-0.3)**: Activities with some impact
            - ⚪ **Weak correlation (<0.1)**: Less impact on success
            """)
        else:
            st.info("Activity correlation data not available.")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888;'>🎓 Skill Gap Awareness System | "
        "Powered by Machine Learning | Built with Streamlit</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
