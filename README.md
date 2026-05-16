## Abstract
I was learning how to build a machine learning model. I learned how a linear one worked, I asked my friend for a mathematical formula, and I got `y = -17x^4 + x^3 + 83x - 17`. Then, I started to code a machine learning model specifically for it with a linear model, found some errors and incompatibilities, solved the issues, and there you go.

I documented all the mistakes I made, funny stuff, and interesting things I found throughout development in the Python code. Have fun reading :)

Fun Fact: I only knew how TypeScript and JavaScript work, and I had no prior knowledge of Python.

## Structure
Very simple, `model.py` is the model code, and `training_data.pth` is the path configuration file that contains two important things: the weightings and the attempts. You may ignore the other files.

## How to run it
1. Clone it
2. Open the folder in a code editor
3. Open the terminal
4. Run `python model.py`
5. Check the outputs.

## What does the output mean?
- Epoch means the data point; you can take it as an attempt.
- Loss value is how close the output is to the actual answer; the bigger it is, the more inaccurate it is.
