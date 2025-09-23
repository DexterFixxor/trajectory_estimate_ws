import torch

all_configs = {
    ### lunar-cosmos-5
    "best": dict(
        window_size_input = 10,
        k = 1,
        window_size_output = 10,
        num_lstm_layers= 1,
        hidden_lstm_neurons=256,
        ff_hidden= [],
        optimizer_class ="AdamW",
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        # data_path = "/home/dexter/Programming/Python/Jefimija/VICON_Tracking/src/data.pth",
        lr = 1e-3,
        batch_size = 256,
        n_epochs = 20,
    ),
}
