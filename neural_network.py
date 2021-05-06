import pickle
import numpy as np
import random


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, np.zeros(z.shape))


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return np.array([1 if cz > 0 else 0 for cz in z])


def cost(output: np.ndarray, label: np.ndarray) -> float:
    return sum(np.power(output - label, 2)) / 2


def cost_derivative(output: np.ndarray, label: np.ndarray) -> np.ndarray:
    return output - label


def check(output: np.ndarray, label: np.ndarray) -> bool:
    return output.argmax() == label.argmax()


class Network:
    def __init__(self, layers: list[int]):
        self.layers: list[int] = layers
        self.biases: list[np.ndarray] = [np.random.randn(layer_size) for layer_size in layers[1:]]
        self.weights: list[np.ndarray] = [np.array([[random.uniform(-0.3, 0.3) for __ in range(prev_layer_size)] for _ in range(layer_size)]) for layer_size, prev_layer_size in zip(layers[1:], layers[:-1])]

    @staticmethod
    def load(path: str):
        return pickle.load(open(path, "rb"))

    def save(self, path: str):
        pickle.dump(self, open(path, "wb"), pickle.HIGHEST_PROTOCOL)

    def feed_forward(self, activations: np.ndarray) -> np.ndarray:
        for biases, weights in zip(self.biases[:-1], self.weights[:-1]):
            activations = relu(np.dot(weights, activations) + biases)   # todo
        return sigmoid(np.dot(self.weights[-1], activations) + self.biases[-1])

    def train(self, training_data: list[tuple[np.ndarray, np.ndarray]], epochs: int, batch_size: int, learning_rate: float, test_data: list[tuple[np.ndarray, np.ndarray]] = None) -> None:
        for i in range(epochs):
            print("Epoch", i + 1)
            random.shuffle(training_data)
            batches = [training_data[j:j + batch_size] for j in range(0, len(training_data) - (batch_size - 1), batch_size)]
            for j, batch in enumerate(batches):
                print(f"{j}/{len(batches)} batches done", end="\r")
                self.train_batch(batch, learning_rate)
            print("done")
            if test_data:
                self.test(test_data)
            print("\n" * 2)

    def train_batch(self, batch: list[tuple[np.ndarray, np.ndarray]], learning_rate: float) -> None:
        bias_changes = [np.zeros(biases.shape) for biases in self.biases]
        weight_changes = [np.zeros(weights.shape) for weights in self.weights]
        for inputs, label in batch:
            bias_change, weight_change = self.propagate_backwards(inputs, label)
            bias_changes = [b + db for b, db in zip(bias_changes, bias_change)]
            weight_changes = [w + dw for w, dw in zip(weight_changes, weight_change)]
        self.biases = [biases - learning_rate * bias_change for biases, bias_change in zip(self.biases, bias_changes)]
        self.weights = [weights - learning_rate * weight_change for weights, weight_change in zip(self.weights, weight_changes)]

    def propagate_backwards(self, activations: np.ndarray, label: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        bias_change = [np.zeros(biases.shape) for biases in self.biases]
        weight_change = [np.zeros(weights.shape) for weights in self.weights]
        history: list[tuple[np.ndarray, np.ndarray]] = [(activations, activations)]     # z, a
        for biases, weights in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(weights, activations) + biases
            activations = relu(z)   # todo
            history.append((z, activations))
        z = np.dot(self.weights[-1], activations) + self.biases[-1]
        activations = sigmoid(z)
        history.append((z, activations))
        derivatives = cost_derivative(activations, label) * sigmoid_derivative(z)  # influence of last z to cost
        for i in range(1, len(self.layers)):
            bias_change[-i] = derivatives
            weight_change[-i] = np.outer(derivatives, history[-i - 1][1])   # derivatives * prev_a1, derivatives * prev_a2, ...
            derivatives = np.dot(self.weights[-i].transpose(), derivatives) * relu_derivative(history[-i - 1][0])   # todo
        return bias_change, weight_change

    def test(self, test_data: list[tuple[np.ndarray, np.ndarray]], mask: int = None) -> None:
        costs = 0
        corrects = 0
        total = 0
        for inputs, label in filter(lambda tup: tup[1].argmax() == mask, test_data) if mask else test_data:
            output = self.feed_forward(inputs)
            costs += cost(output, label)
            if check(output, label):
                corrects += 1
            total += 1
            print(f"{corrects}/{total} correct - average cost: {costs / total}", end="\r")
        print(f"accuracy: {corrects / total * 100}% - average cost: {costs / total}")
