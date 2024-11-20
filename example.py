import fedn

client = fedn()

current_model = "loaded model"


def train(model):
    print("training")
    return model


def model_train(model):
    newmodel =  train(model)
    return newmodel 

def model_updated(newmodel):
    current_model = newmodel

def client_started():
    # for example serve local flask API with current model
    return

client.init(model_train, model_updated, client_started)

