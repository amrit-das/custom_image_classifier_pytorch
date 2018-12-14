import json

with open("hyper_params.json") as hp:
    data = json.load(hp)
    root_dir = data["root_directory"]
    num_classes = data["num_classes"]
    num_epochs = data["num_epochs"]
    batch_size = data["batch_size"]
    num_workers = data["num_workers"]
    lr = data["learning_rate"]
    optim_name = data["optimizer"] 
    momentum = data["momentum"]
    step_size = data["step_size"]
    gamma = data["gamma"]
print type(str(root_dir)), type(num_classes)
    
