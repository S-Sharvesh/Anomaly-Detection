from graphviz import Digraph

# Create a new directed graph
flowchart = Digraph()

# Define nodes
flowchart.node('A', 'Start')
flowchart.node('B', 'Load Dataset')
flowchart.node('C', 'Preprocess Data')
flowchart.node('D', 'Split Data into Training and Testing Sets')
flowchart.node('E', 'Select Deep Learning Model')

# Define model workflows
flowchart.node('E1', 'Autoencoder')
flowchart.node('F1', 'Train Autoencoder')
flowchart.node('G1', 'Evaluate Reconstruction Loss')
flowchart.node('H1', 'Identify Anomalies (High Loss)')
flowchart.node('I1', 'Generate Output Images (Histograms, Metrics, ROC)')

flowchart.node('E2', 'Variational Autoencoder')
flowchart.node('F2', 'Train VAE')
flowchart.node('G2', 'Evaluate Reconstruction Loss and KL Divergence')
flowchart.node('H2', 'Identify Anomalies (High Loss)')
flowchart.node('I2', 'Generate Output Images (Histograms, Metrics, ROC)')

flowchart.node('E3', 'BiGAN')
flowchart.node('F3', 'Train BiGAN')
flowchart.node('G3', 'Evaluate Generative and Discriminative Loss')
flowchart.node('H3', 'Identify Anomalies')
flowchart.node('I3', 'Generate Output Images (Histograms, Metrics, ROC)')

flowchart.node('E4', 'Seq2Seq Model')
flowchart.node('F4', 'Train Seq2Seq Model')
flowchart.node('G4', 'Evaluate Prediction Errors')
flowchart.node('H4', 'Identify Anomalies')
flowchart.node('I4', 'Generate Output Images (Histograms, Metrics, ROC)')

flowchart.node('E5', 'One-Class SVM')
flowchart.node('F5', 'Train One-Class SVM')
flowchart.node('G5', 'Evaluate Support Vectors')
flowchart.node('H5', 'Identify Anomalies')
flowchart.node('I5', 'Generate Output Images (Histograms, Metrics, ROC)')

flowchart.node('E6', 'LSTM')
flowchart.node('F6', 'Train LSTM')
flowchart.node('G6', 'Evaluate Prediction Errors')
flowchart.node('H6', 'Identify Anomalies')
flowchart.node('I6', 'Generate Output Images (Histograms, Metrics, ROC)')

# Collect outputs
flowchart.node('J', 'Collect Outputs for All Models')
flowchart.node('K', 'Compare Model Performance')
flowchart.node('L', 'End')

# Define edges
flowchart.edges(['AB', 'BC', 'CD', 'DE'])
flowchart.edge('E', 'E1', label='Select Model')
flowchart.edge('E', 'E2')
flowchart.edge('E', 'E3')
flowchart.edge('E', 'E4')
flowchart.edge('E', 'E5')
flowchart.edge('E', 'E6')

# Model workflows
flowchart.edge('E1', 'F1')
flowchart.edge('F1', 'G1')
flowchart.edge('G1', 'H1')
flowchart.edge('H1', 'I1')

flowchart.edge('E2', 'F2')
flowchart.edge('F2', 'G2')
flowchart.edge('G2', 'H2')
flowchart.edge('H2', 'I2')

flowchart.edge('E3', 'F3')
flowchart.edge('F3', 'G3')
flowchart.edge('G3', 'H3')
flowchart.edge('H3', 'I3')

flowchart.edge('E4', 'F4')
flowchart.edge('F4', 'G4')
flowchart.edge('G4', 'H4')
flowchart.edge('H4', 'I4')

flowchart.edge('E5', 'F5')
flowchart.edge('F5', 'G5')
flowchart.edge('G5', 'H5')
flowchart.edge('H5', 'I5')

flowchart.edge('E6', 'F6')
flowchart.edge('F6', 'G6')
flowchart.edge('G6', 'H6')
flowchart.edge('H6', 'I6')

# Collect outputs and compare
flowchart.edges(['I1J', 'I2J', 'I3J', 'I4J', 'I5J', 'I6J'])
flowchart.edge('J', 'K')
flowchart.edge('K', 'L')

# Render the flowchart to a file and display it
flowchart.render('anomaly_detection_flowchart', format='png', cleanup=True)
