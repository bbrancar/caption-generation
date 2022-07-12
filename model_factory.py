
from model import LSTMNetwork


# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    vocab_size = vocab.idx
    my_model = LSTMNetwork(hidden_size, embedding_size, vocab_size, 2, model_type)

    return my_model
