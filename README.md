# SIA TP3

Implementation of perceptron neural network to classify bitmap digits (exercise 3) built with Python

## System requirements

* Python 3.7+

## How to use

* Clone or download this repository in the folder you desire
* In a new terminal, navigate to the `exercise3` repository using `cd`
* When you are ready, enter a command as follows:
```sh
python3 determine_even_or_odd.py | determine_digits_themselves.py <input_data.txt> <config.json> <output_file.csv>
```

### Hyperparameters

The configuration for the algorithm's options is a JSON file with the following structure:

* `"layer_sizes"`: The layer architecture
* `"beta"`: The multiplication factor of the sigmoid activation
* `"learning_rate"`: The multilayer perceptron's learning factor
* `"momentum"`: The momentum factor (0 by default)
* `"error_limit"`: The upper bound of the error

## License

This project is licensed under the MIT License - see the LICENSE file for details.
