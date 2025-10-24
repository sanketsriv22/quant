# Black-Scholes
Black-Scholes Option Pricing Model, with Neural Network fitting

BSM_main.py utilizes the Black Scholes Model class to read dataframes of option pricing data, displaying many theoretical plots

BSM.py stores the class

utils.py stores a bunch of functions needed to both parse the yahoo finance api data and manipulate it with pandas and numpy

neural_network.py is a model network created to learn BSM, and does a really good job. Might be trivial as the basic BSM is deterministic.

Next steps:
Create streamlit GUI, implement greeks and dividends for American options, visualize parameter relationships
