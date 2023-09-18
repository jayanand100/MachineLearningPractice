import numpy as np

def sigmoid_numpy(x):
    return 1/1+np.exp(-x)

def log_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred_new = [max(i, epsilon) for i in y_pred]
    y_pred_new = [min(i, 1-epsilon) for i in y_pred_new]
    y_pred_new = np.array(y_pred_new)

    return -np.mean(y_true * np.log(y_pred_new) + (1-y_true)*(1-np.log(y_pred_new)))

class myNN:
    def __int__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0
        self.loss_threshold = 0.4631

    def fit(self, X, y, epochs, loss_threshold):
        self.gradient_descent(X['age'], X['affordability'], y, epochs, loss_threshold)
        print(print(f"Final weights and bias: w1: {self.w1}, w2: {self.w2}, bias: {self.bias}"))


    def predict(self, X_test):
        weighted_sum = self.w1 * X_test['age'] + self.w2 * X_test['affordibility'] + self.bias
        return sigmoid_numpy(weighted_sum)

    def gradient_descent(self, x1, x2, y_true, epochs, loss_threshold):
        w1 = w2 = 1
        bias = 0
        rate = .5
        n = len(y_true)

        for i in range(epochs):
            weighted_sum = x1*w1 + x2*w2 + bias
            y_predicted = sigmoid_numpy(weighted_sum)
            loss = log_loss(y_true, y_predicted)

            w1d = (1/n)*np.dot(np.transpose(x1), (y_predicted - y_true))
            w2d = (1 / n) * np.dot(np.transpose(x2), (y_predicted - y_true))

            bias_d = np.mean(y_predicted - y_true)

            w1 = w1 - rate*w1d
            w2 = w2 - rate*w2d
            bias = bias - rate*bias_d

            if i %50 == 0:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')

            if loss <= loss_threshold:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break

        return w1, w2, bias