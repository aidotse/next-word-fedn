# Federated Learning for Next-Word Prediction

By AI Sweden Young Talents: Västerås.


# Navigating this codebase

This codebase contains lots of unused experiments. Especially in the ```./train``` folder. The only files/folders used for the final product is 
```./train/LSTM/wordOptLSTM.ipynb```
```./data```
```./client```

## Intro

This project we took in collaboration with Scalout: hence the repo name. 

We were assigned to use their platform for federated learning and select a relevant industry + usecase to make the project in.

The industry we chosed was Mobile devices and the use case was next-word prediction.

A measurable goal we chose was to make it learn a new word.


## Training the foundation model.

All the train notebooks are contained in [./train](train/). We tried many different models but the one we found the best result in was the [LSTM Model](train/LSTM/wordOptLSTM.ipynb)

At first we made our own tokenizer which proved great in the beginning lowering loss and increasing accuarcy. However this made it impossible for it to learn new words and in a good way share the token mappings after they were changed.

Therefore we changed our tokenizer to the BERT one using it's token to id mappings but still using our own embeddnigns.

The model was trained on a singular Nvidia RTX 4080 Super for ~30 minutes.


## Building a UI for testing the models

When we trained and tested the different models we used a simple web UI built in svelte that interacts with a flask API.

## Implementing the Fedn package and platform

When it came to moving this package to something compatible with the scaleout platform things got less straight forward.


### Initial challanges

To begin with and as meantioned we realized that our own word-based tokenization wouldn't fit this usecase as new words would have to be added to the vocabulary.

Therefore we switched to a BERT tokenizer. This also proved better while trainig as we could use more data as the tokens were smaller which highered quality while keeping the base dataset the same.

### Moving project structure

After this we moved all the code in the [LSTM Model](train/LSTM/wordOptLSTM.ipynb) notebook into the fedn structure. This was mostly copy-pasting code and changeing function inputs.

### Unexpected errors

One problem that took a while to figure out the cause was how to save the model paramaters. When training the foundtion model we just saved the "state dict" or event the full model with the code into a .pth file.

But after some debugging we realized that the example fedn project was using the "numpyhelper" and saved the model parameters as a numpy array (.npz)

