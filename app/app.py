
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import pickle
import streamlit_shadcn_ui as ui

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

with open('https://github.com/UmarBalak/Cifar10_VGG/blob/main/app/actual_labels.pkl', 'rb') as f:
    actual_labels = pickle.load(f)
predictions = np.load('https://github.com/UmarBalak/Cifar10_VGG/blob/main/app/predictions.npy')


# Function to calculate confusion matrix
def calculate_confusion_matrix(predictions, actual):
    confusion_matrix = pd.crosstab(actual, predictions.argmax(axis=1), rownames=['Actual'], colnames=['Predicted'])
    confusion_matrix.index = class_names
    confusion_matrix.columns = class_names
    return confusion_matrix

# Function to calculate overall accuracy
def calculate_accuracy(predictions, actual):
    correct_predictions = (predictions.argmax(axis=1) == actual).sum()
    total_predictions = len(actual)
    accuracy = correct_predictions / total_predictions
    return accuracy


# Function to calculate precision, recall, and F1-score per class
def calculate_precision_recall_f1(predictions, actual):
    precision_per_class = precision_score(actual, predictions, average=None)
    recall_per_class = recall_score(actual, predictions, average=None)
    f1_per_class = f1_score(actual, predictions, average=None)
    return precision_per_class, recall_per_class, f1_per_class


def plot_confusion_matrix(confusion_matrix):
    fig = go.Figure(data=go.Heatmap(z=confusion_matrix.values,
                                    x=class_names,
                                    y=class_names,
                                    colorscale='Reds'))
    fig.update_layout(title='Confusion Matrix',
                      xaxis_title='Predicted Label',
                      yaxis_title='Actual Label',
                      template='plotly_white')
    fig.update_traces(hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}', hoverinfo='text',
                      hoverlabel=dict(bgcolor='white', font=dict(color='black')))
    return fig

def plot_accuracy_per_class(confusion_matrix):
    accuracy_per_class = (confusion_matrix.values.diagonal() / confusion_matrix.values.sum(axis=1)) * 100
    fig = go.Figure(go.Bar(x=class_names, y=accuracy_per_class, marker_color='red'))
    fig.update_layout(title='Accuracy per Class',
                      xaxis_title='Class',
                      yaxis_title='Accuracy (%)',
                      template='plotly_white')
    fig.update_traces(hovertemplate='Class: %{x}<br>Accuracy: %{y:.2f}%', hoverinfo='text',
                      hoverlabel=dict(bgcolor='white', font=dict(color='black')))
    return fig

def plot_precision_recall_f1(precision_per_class, recall_per_class, f1_per_class):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=class_names, y=precision_per_class, mode='lines+markers', name='Precision', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=class_names, y=recall_per_class, mode='lines+markers', name='Recall', line=dict(color='mediumseagreen')))
    fig.add_trace(go.Scatter(x=class_names, y=f1_per_class, mode='lines+markers', name='F1-score', line=dict(color='salmon')))
    fig.update_layout(title='Precision, Recall, and F1-score per Class',
                      xaxis_title='Class',
                      yaxis_title='Metric Value',
                      template='plotly_white')
    fig.update_traces(hovertemplate='Class: %{x}<br>Metric: %{y:.4f}',
                      hoverlabel=dict(bgcolor='red', font=dict(color='white')))
    return fig

def plot_layer_distribution():
    layers = ['Conv2D', 'MaxPooling2D', 'Dense', 'BatchNormalization', 'Dropout']
    layer_counts = [6, 3, 2, 7, 4]
    total_layers = sum(layer_counts)
    percentages = [count / total_layers * 100 for count in layer_counts]

    text = [f'{layer}: {count}' for layer, count, percentage in zip(layers, layer_counts, percentages)]

    fig = go.Figure(data=[go.Pie(labels=text, values=layer_counts, textinfo='label+percent')])
    fig.update_layout(title='Layer Distribution in the Model', template='plotly_white')
    fig.update_traces(hoverinfo='label+percent',
                      hoverlabel=dict(bgcolor='white', font=dict(color='black')))
    return fig

def diff_layer_dist(categories, values1, values2):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories,
                         y=values1,
                         name='Cifar-VGG',
                         marker_color='rgb(55, 83, 109)'
                         ))
    fig.add_trace(go.Bar(x=categories,
                         y=values2,
                         name='TinyVGG',
                         marker_color='rgb(26, 118, 255)'
                         ))

    fig.update_layout(
        title='CNN Layer Distribution',
        xaxis_tickfont_size=14,
        xaxis = dict(
            title='Layer Name',
        ),
        yaxis=dict(
            title='Layer Count',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0.9,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            traceorder='reversed'  
        ),
        barmode='group',
        bargap=0.15,  
        bargroupgap=0.07  
    )
    return fig

# Calculate confusion matrix
confusion_matrix = calculate_confusion_matrix(predictions, actual_labels)

# Calculate overall accuracy
accuracy = calculate_accuracy(predictions, actual_labels)

# Calculate precision, recall, and F1-score per class
precision_per_class, recall_per_class, f1_per_class = calculate_precision_recall_f1(predictions.argmax(axis=1), actual_labels)

# Calculate layer distribution
layer_distribution = plot_layer_distribution()

diff_layer = diff_layer_dist(['Conv2D', 'Maxpool2D', 'Normalization', 'Dropout', 'Dense'], [13, 5, 14, 10, 2], [6, 3, 7, 4, 2])

def main(confusion_matrix, accuracy, precision_per_class, layer_distribution):
    st.set_page_config(layout="wide", page_title="Model Performance Dashboard", page_icon="📊")

    st.write("<style>.option-menu-container { margin-top: -30px; }</style>", unsafe_allow_html=True)
    page = option_menu(
        menu_title=None,
        options=["Home", "Architecture", "Performance", "Compare"],
        icons = ["house", "diagram-2-fill", "bar-chart", "activity"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
    )

    if page == "Home":
        st.write("<div style='margin: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #000000;'>TinyVGG</h2>", unsafe_allow_html=True)
        
        cols = st.columns((1, 7, 1))
        with cols[1]:
            st.markdown('''<p style='text-align: justify;'>Get ready to dive into the exciting world of image classification with Convolutional Neural Networks (CNNs)!
                     Explore the model's performance metrics and unleash the power of deep learning on the Dashboard tab.
                    Navigate to the Dashboard tab now to witness the magic!
                    </p>''', unsafe_allow_html=True)
            st.write("<hr>", unsafe_allow_html=True)


            st.markdown("<h4 style='color: #FF4B4B;'>Problem Statement</h4>", unsafe_allow_html=True)
            st.markdown('''<p style='text-align: justify;'>The aim of this project was to enhance the performance of image classification using
                        a customized Convolutional Neural Network (CNN) architecture tailored for the CIFAR-10 dataset, surpassing the capabilities
                        of the existing VGG-16 architecture. Three primary objectives guided this endeavor:
                        Firstly, to bolster the accuracy of classification results, ensuring more precise and reliable predictions across diverse classes within the CIFAR-10 dataset. 
                        Secondly, to mitigate the computational burden associated with training by reducing the overall training time required to achieve optimal performance,
                        thereby enhancing the efficiency of the model development process. Lastly, to address the issue of model size, aiming to create a more compact architecture without sacrificing accuracy, 
                        thereby facilitating easier deployment and integration into practical applications with constrained resources. These objectives collectively aimed to push the boundaries of image classification performance 
                        while simultaneously addressing critical challenges such as computational complexity and model size, thus paving the way for more effective and efficient CNN-based solutions in real-world scenarios.</p>''',
                        unsafe_allow_html=True)
            
    elif page == "Architecture":
        st.write("<div style='margin: 10px;'></div>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #000000;'>TinyVGG Architecture</h2>", unsafe_allow_html=True)
        cols = st.columns((1, 7, 1))
        with cols[1]:
            st.markdown('''<p style='text-align: justify;'>
                        The TinyVGG architecture is a convolutional neural network (CNN) model that strikes a balance between simplicity and effectiveness in image classification tasks. 
                        This architecture consists of a total of 8 layers, including 6 convolutional layers and 2 fully connected layers.
                            At the beginning of the network, the input layer accepts images of size 32x32 pixels with 3 color channels (RGB). 
                        The convolutional layers extract features from the input images, each followed by a rectified linear unit (ReLU) activation function. 
                        Max pooling layers are applied after every two convolutional layers to reduce the spatial dimensions of the feature maps using 2x2 filters with a stride of 2. 
                        The final two layers of the architecture are fully connected layers. The first fully connected layer has 256 neurons, followed by a second fully connected layer with 10 neurons,
                            corresponding to the 10 classes in the CIFAR-10 dataset, on which the architecture was originally trained. 
                        The output layer of the network uses a softmax activation function to output class probabilities for the input image. 
                        The architecture is implemented using the TensorFlow/Keras framework,
                            with additional layers such as batch normalization and dropout included to improve model performance and generalization.
                        </p>''', unsafe_allow_html=True)
            
            st.write("<div style='margin: 40px;'></div>", unsafe_allow_html=True)
            image_url = 'TinyVGG-Architecture.png'
            st.image(image_url, caption='TinyVGG Architecture', use_column_width=True)



    elif page == "Performance":
        st.markdown("<h2 style='text-align: center; color: #000000;'>Model Performance</h2>", unsafe_allow_html=True)
        st.write("<div style='margin: 40px;'></div>", unsafe_allow_html=True)

        fig_confusion_matrix = plot_confusion_matrix(confusion_matrix)
        fig_accuracy_per_class = plot_accuracy_per_class(confusion_matrix)
        fig_precision_recall_f1 = plot_precision_recall_f1(precision_per_class, recall_per_class, f1_per_class)

        cols = st.columns(4)
        with cols[0]:
            ui.metric_card(
                title="Accuracy",
                content="91.94%",
                description="-1.49%",
                key = "card1"
            )
        with cols[1]:
            ui.metric_card(
                title="Loss",
                content="0.39",
                description="-0.08",
                key = "card2"
            )
        with cols[2]:
            ui.metric_card(
                title="Size",
                content="4.22 MB",
                description="-53.01.MB",
                key = "card3"
            )
        with cols[3]:
            ui.metric_card(
                title="Parameters",
                content="1086026",
                description="-13915392",
                key = "card4"
            )
        
        st.write("<hr>", unsafe_allow_html=True)


        st.markdown(
            f"""
            <style>
            .stPlotlyChart {{
            outline: 15px solid {"#FFFFFF"};
            border-radius: 3px;
            box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.20), 0 6px 20px 0 rgba(0, 0, 0, 0.30);
            }}
            </style>
            """, unsafe_allow_html=True
        )
                
        left, m,  right = st.columns([0.48, 0.04, 0.48])
        with left:
            st.plotly_chart(fig_confusion_matrix, use_container_width=True)
        with right:
            st.plotly_chart(fig_accuracy_per_class, use_container_width=True)

        st.write("<div style='margin: 40px;'></div>", unsafe_allow_html=True)
        left, m, right = st.columns([0.43, 0.04, 0.53])
        with left:
            st.plotly_chart(layer_distribution, use_container_width=True)
        with right:
            st.plotly_chart(fig_precision_recall_f1, use_container_width=True)



    elif page == "Compare":
        st.markdown("<h2 style='text-align: center; color: #000000;'>TinyVGG &nbsp; vs &nbsp; Cifar-VGG</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: #000000;'>Exploring the Depths of Performance and Accuracy.</h4>", unsafe_allow_html=True)

        st.write("<div style='margin: 40px;'></div>", unsafe_allow_html=True)

        css = """
        <style>
        .subheader-color {
            color: #FF4B4B; /* Specify your desired color */
        }
        </style>
        """
        
        st.write(css, unsafe_allow_html=True)

        e1, c1, e2 = st.columns((1, 7, 1))

        with c1:
            st.markdown("<h4 class='subheader-color'>Cifar-VGG</h4>", unsafe_allow_html=True)
            c11, c12, c21, c22 = st.columns(4)
            with c11:
                ui.metric_card(
                    title="Accuracy",
                    content="93.43%",
                    key="metric_card_1_1"
                )
            with c12:
                ui.metric_card(
                    title="Loss",
                    content="0.47",
                    key="metric_card_1_2"
                )
            with c21:
                ui.metric_card(
                    title="Model Size",
                    content="57.23 MB",
                    key="metric_card_2_1"
                )
            with c22:
                ui.metric_card(
                    title="Parameters",
                    content="15001418",
                    key="metric_card_2_2"
                )
        
        st.write("<div style='margin: 10px;'></div>", unsafe_allow_html=True)
        e1, c1, e2 = st.columns((1, 7, 1))
        with c1:
            st.markdown("<h4 class='subheader-color'>TinyVGG</h4>", unsafe_allow_html=True)
            c11, c12, c21, c22 = st.columns(4)
            with c11:
                ui.metric_card(
                    content="91.94%",
                    description="-1.49%",
                    key="metric_card_3_1"
                )
            with c12:
                ui.metric_card(
                    content="0.39",
                    description="-0.08",
                    key="metric_card_3_2"
                )
            with c21:
                ui.metric_card(
                    content="4.22 MB",
                    description="-53.01.MB",
                    key="metric_card_4_1"
                )
            with c22:
                ui.metric_card(
                    content="1086026",
                    description="-13915392",
                    key="metric_card_4_2"
                )
       
        st.write("<hr>", unsafe_allow_html=True)

        st.markdown(
                f"""
                <style>
                .stPlotlyChart {{
                outline: 15px solid {"#FFFFFF"};
                border-radius: 3px;
                box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.20), 0 6px 20px 0 rgba(0, 0, 0, 0.30);
                }}
                </style>
                """, unsafe_allow_html=True
            )
                    
        left, m,  right = st.columns([0.15, 0.7, 0.15])
        with m:
            st.plotly_chart(diff_layer, use_container_width=True)

        st.write("<div style='margin: 40px;'></div>", unsafe_allow_html=True)
       
        st.markdown("<h4 class='subheader-color'>Optimizer</h4>", unsafe_allow_html=True)
        c11, c12 = st.columns(2)
        with c11:
            code = '''def optimizer():
        return keras.optimizers.SGD()'''
            st.code(code, language='python')
        with c12:
            code = '''def optimizer():
        return keras.optimizers.Adam()'''
            st.code(code, language='python')

if __name__ == "__main__":
    main(confusion_matrix, accuracy, precision_per_class, layer_distribution)
