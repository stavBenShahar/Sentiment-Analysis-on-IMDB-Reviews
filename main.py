import torch
from torch.nn.functional import pad
import torch.nn as nn
import numpy as np
import loader as ld
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from tqdm import tqdm
from loader import preprocess_review
import torch.nn.functional as F
import math

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing device: {device}')

batch_size = 32

# Loading dataset, use toy=True for obtaining a smaller dataset
train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)


class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias=True):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels, out_channels)),
                                         requires_grad=True)
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(1, 1, out_channels), requires_grad=True)

        self.use_bias = use_bias

    def forward(self, x):
        x = torch.matmul(x, self.matrix)
        if self.use_bias:
            x = x + self.bias
        return x


class ExRNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRNN, self).__init__()

        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid

        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden2out = nn.Linear(hidden_size, output_size)

    @staticmethod
    def name():
        return "RNN"

    def forward(self, x, hidden_state):
        hidden = self.sigmoid(self.in2hidden(torch.cat((x, hidden_state), 1)))
        output = self.hidden2out(hidden)
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size).to(device)


class ExGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExGRU, self).__init__()
        self.hidden_size = hidden_size
        self.sigmoid = torch.sigmoid
        self.Wz = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wr = nn.Linear(input_size + hidden_size, hidden_size)
        self.W = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wy = nn.Linear(hidden_size, output_size)

    @staticmethod
    def name():
        return "GRU"

    def forward(self, x, hidden_state):
        z = self.Wz(torch.concat((x, hidden_state), 1))
        r = self.Wr(torch.concat((x, hidden_state), 1))
        h = nn.Tanh()(self.W(torch.concat((r * hidden_state, x), 1)))
        hidden = self.sigmoid((1 - z) * hidden_state + z * h)
        output = self.Wy(hidden)
        return output, hidden

    def init_hidden(self, bs):
        return torch.zeros(bs, self.hidden_size).to(device)


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()

        self.layer1 = MatMul(input_size, hidden_size)
        self.layer2 = MatMul(hidden_size, output_size)
        self.ReLU = torch.nn.ReLU()

    @staticmethod
    def name():
        return "MLP"

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)
        x = self.ReLU(x)
        return x


class ExRestSelfAtten(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExRestSelfAtten, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.sqrt_hidden_size = np.sqrt(float(hidden_size))
        self.ReLU = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

        self.layer1 = MatMul(input_size, hidden_size).to(device)
        self.W_q = MatMul(hidden_size, hidden_size, use_bias=False).to(device)
        self.W_k = MatMul(hidden_size, hidden_size, use_bias=False).to(device)
        self.W_v = MatMul(hidden_size, hidden_size, use_bias=False).to(device)
        self.layer2 = MatMul(hidden_size, output_size).to(device)

    @staticmethod
    def name():
        return "MLP_atten"

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(x)

        padded = pad(x, (0, 0, atten_size, atten_size, 0, 0))

        x_nei = []
        for k_T in range(-atten_size, atten_size + 1):
            x_nei.append(torch.roll(padded, k_T, 1))

        x_nei = torch.stack(x_nei, 2).to(device)
        x_nei = x_nei[:, atten_size:-atten_size, :]

        query = self.W_q(x)
        keys = self.W_k(x_nei)
        values = self.W_v(x_nei)

        # Calculate attention scores
        attention_scores = torch.matmul(query.unsqueeze(2), keys.transpose(-2, -1)).squeeze(2) / self.sqrt_hidden_size
        attention_weights = self.softmax(attention_scores)

        # Calculate the weighted sum of the values
        attention_output = torch.matmul(attention_weights.unsqueeze(2), values).squeeze(2)

        x = self.layer2(attention_output)
        x = self.ReLU(x)

        return x, attention_weights



def print_review(rev_text, sub_scores,true_labels):
    """
    Prints a portion of the review (first 30 words), with the sub-scores each word obtained.
    Prints also the final scores, the softmaxed prediction values, and the true label values.

    Args:
        rev_text (str): The review text.
        sub_scores (torch.Tensor): Sub-scores for each word.
        true_labels (torch.Tensor): True label values.
    """
    words = rev_text.split()[:30]
    review_with_scores = ["Review (first 30 words):"]
    sub_scores = sub_scores.squeeze().squeeze()
    # Print the words with their corresponding sub-scores
    final_scores = torch.Tensor([0, 0])
    for i, word in enumerate(words):
        score = sub_scores[i]
        final_scores += score
        review_with_scores.append(f"{word} ({score})")
    review_with_scores.append("\n")

    # Print the final scores
    review_with_scores.append(f"Final Scores (before softmax): {final_scores}\n")

    # Print the softmax prediction values
    review_with_scores.append(f"Softmax Prediction Values: {torch.softmax(final_scores,dim=0)}\n")

    # Print the true label values
    review_with_scores.append(f"True Label Values: {true_labels}\n")

    return '\n'.join(review_with_scores)


def accuracy(y_pred, y_true):
    """
    Calculate the accuracy of the model's predictions.
    """
    size = y_true.shape[0]
    y_pred = torch.softmax(y_pred, dim=1)
    rounded_y_pred = torch.round(y_pred)
    agree = y_true == rounded_y_pred
    val = sum([1 for i in range(size) if agree[i][0] == agree[i][1] == True])
    percentage = 100 * float(val) / size
    return percentage


def choose_model():
    """
    Choose and initialize the model based on the configuration settings.
    """
    if run_recurrent:
        chosen_model = ExRNN(input_size, output_size, hidden_size) if use_RNN else ExGRU(input_size, output_size,
                                                                                         hidden_size)
    else:
        chosen_model = ExRestSelfAtten(input_size, output_size, hidden_size) if atten_size > 0 else ExMLP(input_size,
                                                                                                          output_size,
                                                                                                          hidden_size)
    print(f"Using model: {chosen_model.name()}")
    return chosen_model.to(device)


def forward_pass(model, labels, reviews, criterion=None):
    """
    Perform a forward pass through the model.
    """
    global sub_score, output
    if run_recurrent:
        hidden_state = model.init_hidden(labels.size(0))
        sub_scores = []
        for i in range(num_words):
            sub_score, hidden_state = model(reviews[:, i, :].to(device), hidden_state)
            sub_scores.append(sub_score)
        sub_scores = torch.stack(sub_scores, dim=1)
        output = torch.sum(sub_scores, dim=1)
    else:
        if atten_size > 0:
            sub_score, atten_weights = model(reviews.to(device))
        else:
            sub_score = model(reviews.to(device))
        output = torch.sum(sub_score, dim=1)

    if criterion:
        loss = criterion(output, labels.to(device))
        return (loss, output) if run_recurrent else (loss, output, sub_score)
    else:
        return output if run_recurrent else (output, sub_score)


def train_step(model, train_dataset, optimizer):
    """
    Perform a single training step.
    """
    model.train()
    train_loss, train_accuracy = 0, 0
    total_batches = len(train_dataset)
    with tqdm(total=total_batches, desc="Training", unit="batch") as pbar:
        for cur_train_batch, (train_labels, train_reviews, _) in enumerate(train_dataset, 1):
            if run_recurrent:
                loss, train_output = forward_pass(model, train_labels, train_reviews)
            else:
                loss, train_output, _ = forward_pass(model, train_labels, train_reviews)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += accuracy(train_output, train_labels.to(device))

            pbar.set_postfix({
                'Avg Loss': f'{train_loss / cur_train_batch:.4f}',
                'Avg Accuracy': f'{train_accuracy / cur_train_batch:.2f}%'
            })
            pbar.update(1)

    train_loss /= total_batches
    train_accuracy /= total_batches

    return train_loss, train_accuracy


def evaluate_step(model, test_dataset):
    """
    Evaluate the model on the test dataset.
    """
    model.eval()
    test_loss, test_accuracy = 0, 0
    total_batches = len(test_dataset)
    with torch.no_grad():
        with tqdm(total=total_batches, desc="Evaluating", unit="batch") as pbar:
            for cur_test_batch, (test_labels, test_reviews, _) in enumerate(test_dataset, 1):
                if run_recurrent:
                    loss, test_output = forward_pass(model, test_labels, test_reviews)
                else:
                    loss, test_output, _ = forward_pass(model, test_labels, test_reviews)

                test_loss += loss.item()
                test_accuracy += accuracy(test_output, test_labels.to(device))

                pbar.set_postfix({
                    'Avg Loss': f'{test_loss / cur_test_batch:.4f}',
                    'Avg Accuracy': f'{test_accuracy / cur_test_batch:.2f}%'
                })
                pbar.update(1)

    test_loss /= total_batches
    test_accuracy /= total_batches

    return test_loss, test_accuracy


def display_plot(train_accuracy_arr, test_accuracy_arr, model_name, num_epochs):
    """
    Display and save a plot of training and testing accuracy over epochs.
    """
    plt.figure(figsize=(10, 5))

    plt.plot(range(1, num_epochs + 1), train_accuracy_arr, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracy_arr, label='Test Accuracy')

    title = f"Accuracy over Epochs [Model {model_name}] [HiddenSize {hidden_size}]"
    plt.title(title)
    plt.xlabel("Epochs")
    plt.xlim([1, num_epochs])
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(os.path.join('plots', title))
    plt.show()


def print_epoch_metrices(train_loss, train_acc, test_loss, test_acc):
    """
    Print training and testing metrics for each epoch.
    """
    print(
        f"Train Loss: {train_loss:.2f}, "
        f"Train Accuracy: {train_acc:.2f}%, "
        f"Test Loss: {test_loss:.2f}, "
        f"Test Accuracy: {test_acc:.2f}%"
    )


def initialize_model(save_directory="trained models"):
    """
    Initialize and optionally reload the model.
    """

    model = choose_model()
    if reload_model:
        print(f"Reloading model - {model.name} {hidden_size}")
        model.load_state_dict(
            torch.load(os.path.join(save_directory, f"{model.name()} [HiddenSize {hidden_size}].pth")))
    return model


def define_loss_and_optimizer(model, learning_rate):
    """
    Define the loss function and optimizer.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer


def train_and_evaluate_model(model, train_dataset, test_dataset, optimizer, num_epochs, hidden_size):
    """
    Train and evaluate the model for a specified number of epochs.
    """
    print(f'Current hidden size: {hidden_size}\n')
    train_accuracy_arr, test_accuracy_arr = [], []
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        train_loss, train_accuracy = train_step(model, train_dataset, optimizer)
        train_accuracy_arr.append(train_accuracy)

        test_loss, test_accuracy = evaluate_step(model, test_dataset)
        test_accuracy_arr.append(test_accuracy)

        print_epoch_metrices(train_loss, train_accuracy, test_loss, test_accuracy)

    return train_accuracy_arr, test_accuracy_arr


def predict_sentiments(model, reviews, true_labels, output_file="test_reviews.txt"):
    """
    Predict sentiments for a list of reviews using the provided model and save the results to a file.

    Args:
        model (nn.Module): The trained model.
        reviews (list): A list of reviews to predict sentiments for.
        true_labels (list): The true labels for the reviews.
        output_file (str): The path to the output file.
    """
    model.eval()  # Set the model to evaluation mode
    sentiments = []

    with torch.no_grad():  # Disable gradient calculation
        for i, review in enumerate(reviews):
            processed_review = preprocess_review(review).to(device)
            labels = torch.tensor([ld.preprocess_label(true_labels[i])], dtype=torch.float32).to(device)

            # Ensure processed_review has the correct shape
            if run_recurrent:
                processed_review = processed_review.squeeze(0)  # Remove the batch dimension
                if processed_review.dim() == 2:
                    processed_review = processed_review.unsqueeze(0)  # Add batch dimension back
                processed_review = processed_review.expand(1, num_words, ld.embedding_size)
            else:
                processed_review = processed_review.unsqueeze(0)

            # Perform a forward pass
            if run_recurrent:
                hidden_state = model.init_hidden(labels.size(0))
                sub_scores = []
                for j in range(num_words):
                    sub_score, hidden_state = model(processed_review[:, j, :].to(device), hidden_state)
                    sub_scores.append(sub_score)
                sub_scores = torch.stack(sub_scores, dim=1)
                output = torch.sum(sub_scores, dim=1)
            else:
                if atten_size > 0:
                    sub_score, atten_weights = model(processed_review.to(device).squeeze())
                else:
                    sub_score = model(processed_review.to(device))
                output = sub_score.squeeze().squeeze().sum(dim=0)

            # Decode the output to get the sentiment
            prediction = torch.argmax(output).item()
            sentiment = "positive" if prediction == 1 else "negative"
            sentiments.append((review, sentiment, sub_score, output, F.softmax(output,dim=0), labels))

    with open(output_file, 'a') as f:
        f.write(f"\n{model.name()}\n")
        for review, sentiment, sub_score, final_score, softmax_preds, true_label in sentiments:
            if run_recurrent:
                f.write(f"{review} | {sentiment}\n")
            else:
                review_with_scores = print_review(review, sub_score, true_label)
                f.write(f"{review_with_scores}\n")


def set_model_configuration(model_string):
    """
    Set the configuration for use_RNN, run_recurrent, and atten_size based on the model string.

    Args:
        model_string (str): The string representing the desired model.
                           Possible values: 'RNN', 'GRU', 'MLP', 'MLP_atten'
    """
    global use_RNN, run_recurrent, atten_size

    if model_string == 'RNN':
        use_RNN = True
        run_recurrent = True
        atten_size = 0
    elif model_string == 'GRU':
        use_RNN = False
        run_recurrent = True
        atten_size = 0
    elif model_string == 'MLP':
        use_RNN = False
        run_recurrent = False
        atten_size = 0
    elif model_string == 'MLP_atten':
        use_RNN = False
        run_recurrent = False
        atten_size = 5  # You can set this to any desired attention size
    else:
        raise ValueError("Invalid model string. Choose from 'RNN', 'GRU', 'MLP', 'MLP_atten'.")


def save_model(model, save_directory="trained models"):
    """
    Save the model state dictionary.
    """
    os.makedirs(save_directory, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_directory, f"{model.name()} [HiddenSize {hidden_size}].pth"))


if __name__ == '__main__':
    num_epochs = 10
    output_size = 2
    learning_rate = 0.001
    hidden_sizes = [64, 96, 128]
    reload_model = True

    # Need to train the models
    if not reload_model:
        models = ['RNN', 'GRU', 'MLP', 'MLP_atten']
        for model in models:
            set_model_configuration(model)
            for hidden_size in hidden_sizes:
                model = initialize_model()
                criterion, optimizer = define_loss_and_optimizer(model, learning_rate)
                train_accuracy_arr, test_accuracy_arr = train_and_evaluate_model(
                    model, train_dataset, test_dataset, optimizer, num_epochs, hidden_size
                )
                display_plot(train_accuracy_arr, test_accuracy_arr, model.name(), num_epochs)
                save_model(model)
    else:
        hidden_size = 128
        #Set the model configuration for RNN
        # set_model_configuration('RNN')
        # trained_model_rnn = initialize_model()
        # predict_sentiments(trained_model_rnn, ld.my_test_texts, ld.my_test_labels)
        # hidden_size = 64
        # # Set the model configuration for GRU
        # set_model_configuration('GRU')
        # trained_model_gru = initialize_model()
        # predict_sentiments(trained_model_gru, ld.my_test_texts,  ld.my_test_labels)
        # hidden_size = 128
        # Set the model configuration for MLP
        set_model_configuration('MLP')
        trained_model_mlp = initialize_model()
        predict_sentiments(trained_model_mlp, ld.my_test_texts, ld.my_test_labels)
        hidden_size = 96
        # Set the model configuration for MLP_atten
        set_model_configuration('MLP_atten')
        trained_model_mlp_atten = initialize_model()
        predict_sentiments(trained_model_mlp_atten, ld.my_test_texts, ld.my_test_labels)
