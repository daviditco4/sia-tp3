import subprocess

# Define the command template and the arguments
command_template = 'python3'
arguments = [
    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/prototype.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/hidden_layer_3.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/hidden_layer_8.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/hidden_layer_15.json exercise3/outputs/classification_even_or_odd.csv',

    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/beta_0-5.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/beta_1-5.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/beta_2.json exercise3/outputs/classification_even_or_odd.csv',

    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/learning_rate_0-2.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/learning_rate_0-5.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/learning_rate_1.json exercise3/outputs/classification_even_or_odd.csv',

    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/momentum_0-5.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/momentum_0-9.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_even_or_odd.py exercise3/data/digits.txt exercise3/configs/momentum_0-99.json exercise3/outputs/classification_even_or_odd.csv',


    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_prototype.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_hidden_layer_5.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_hidden_layer_8.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_hidden_layer_20.json exercise3/outputs/classification_even_or_odd.csv',

    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_beta_0-5.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_beta_1-5.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_beta_2.json exercise3/outputs/classification_even_or_odd.csv',

    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_learning_rate_0-2.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_learning_rate_0-5.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_learning_rate_1.json exercise3/outputs/classification_even_or_odd.csv',

    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_momentum_0-5.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_momentum_0-9.json exercise3/outputs/classification_even_or_odd.csv',
    'exercise3/determine_digits_themselves.py exercise3/data/digits.txt exercise3/configs/digit_determination_momentum_0-99.json exercise3/outputs/classification_even_or_odd.csv',
]

# Number of repetitions
num_repetitions = 10


def run_command(command, args, repetitions):
    for arg in args:
        for _ in range(repetitions):
            # Construct the full command
            full_command = [command] + arg.split()
            try:
                # Execute the command
                result = subprocess.run(full_command, capture_output=True, text=True, check=True)
                print(f'Command executed: {result.args}')
                print('Output:')
                print(result.stdout.strip())
            except subprocess.CalledProcessError as e:
                print(f'Error executing command: {e}')
                print(f'Command: {e.cmd}')
                print(f'Output: {e.output}')
                print(f'Error: {e.stderr}')


if __name__ == "__main__":
    run_command(command_template, arguments, num_repetitions)
