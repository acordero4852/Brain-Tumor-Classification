import plotly.graph_objects as go
import streamlit as st


# Create the result container to display the prediction and confidence
def create_result_container(result, prediction, class_index):
    result_container = st.container()
    result_container = st.container()
    result_container.markdown(
        f"""
          <div style="background-color: #000000; color: #ffffff; padding: 30px; border-radius: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
              <div style="flex: 1; text-align: center;">
                <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Prediction</h3>
                <p style="font-size: 36px; font-weight: 800; color: #52ff76; margin: 0;">
                  {result}
                </p>
              </div>
              <div style="width: 2px; height: 80px; background-color: #ffffff; margin: 0 20px;"></div>
              <div style="flex: 1; text-align: center;">
                <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Confidence</h3>
                <p style="font-size: 36px; font-weight: 800; color: #00b4d8; margin: 0;">
                  {prediction[0][class_index]:.4%}
                </p>
              </div>
            </div>
          </div>
        """,
        unsafe_allow_html=True,
    )


# Create a probabilities chart to display the prediction probabilities
def create_probabilities_chart(result, sorted_probabilities, sorted_labels):
    fig = go.Figure(
        go.Bar(
            x=sorted_probabilities,
            y=sorted_labels,
            orientation="h",
            marker_color=[
                "#52ff76" if label == result else "#00b4d8" for label in sorted_labels
            ],
        )
    )

    fig.update_layout(
        title="Probabilities",
        xaxis_title="Probability",
        yaxis_title="Class",
        height=400,
        width=6000,
        yaxis=dict(autorange="reversed"),
    )

    for i, prob in enumerate(sorted_probabilities):
        fig.add_annotation(
            x=prob, y=i, text=f"{prob:.4f}", showarrow=False, xanchor="left", xshift=5
        )

    return fig
