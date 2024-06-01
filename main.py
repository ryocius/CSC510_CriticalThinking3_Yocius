import numpy as np

sequence1 = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]])
sequence2 = np.array([[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15],[16]])

# Input Layer
inSize = sequence1.shape[1]

# Hidden Layer
hiddenSize = 4

# Output Layer
outSize = 1

# Randomly initialized weights
WHidden = np.random.randn(inSize, hiddenSize)
bHidden = np.zeros((1, hiddenSize))
WOut = np.random.randn(hiddenSize, outSize)
bOut = np.zeros((1, outSize))

def sigmoid(sequence1):
    return (1 / (1 + np.exp(-sequence1)))

# Train the ANN
for epoch in range(2000):
    # Forward pass / Feedforward
    hiddenLayerIn = np.dot(sequence1, WHidden) + bHidden
    hiddenLayerOut = sigmoid(hiddenLayerIn)
    outLayerIn = np.dot(hiddenLayerOut, WOut) + bOut
    predOut = sigmoid(outLayerIn)

    # Compute loss (mean squared error)
    loss = np.mean((predOut - sequence2) ** 2)

    # Backpropagation
    deltaOut = (predOut - sequence2) * predOut * (1 - predOut)
    deltaHidden = np.dot(deltaOut, WOut.T) * hiddenLayerOut * (1 - hiddenLayerOut)

    # Update weights and biases
    learnRate = 0.01
    WOut -= learnRate * np.dot(hiddenLayerOut.T, deltaOut)
    bOut -= learnRate * np.sum(deltaOut, axis=0)
    WHidden -= learnRate * np.dot(sequence1.T, deltaHidden)
    bHidden -= learnRate * np.sum(deltaHidden, axis=0)


# Print final predicted output
print("Predicted output strengths after training:", predOut)


def predictIfNext(X, inVal):
    newIn = np.array([[inVal]])
    Xextended = np.vstack((X, newIn))

    # Compute forward pass for the extended sequence
    hiddenLayerIn_extended = np.dot(Xextended, WHidden) + bHidden
    hiddenLayerOut_extended = sigmoid(hiddenLayerIn_extended)
    outLayerIn_extended = np.dot(hiddenLayerOut_extended, WOut) + bOut
    predOut_extended = sigmoid(outLayerIn_extended)

    return predOut_extended[-1]

# Using the weights, guess the next value in the sequence1 Array
nextVal = 0
weights = []
for i in range(0, 100):
    weights.append({'index': i, 'value': predictIfNext(sequence1, i).item()})

maxIndex = max(weights, key=lambda x: x['value'])['index']
out = str(sequence1).replace('\n', '')

print(f"Based on the trained model, the next value in the array {out} is most likely {maxIndex}" )
