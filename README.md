## Abstract
I was learning how to build a machine learning model. I learned how a linear one worked, I asked my friend for a mathematical formula, and I got `y = -17x^4 + x^3 + 83x - 17`. Then, I started to code a machine learning model specifically for it with a linear model, found some errors and incompatibilities, solved the issues, and there you go.

I documented all the mistakes I made, funny stuff, and interesting things I found throughout development in the Python code. Have fun reading :)

Fun Fact: I only knew how TypeScript and JavaScript work, and I had no prior knowledge of Python.

## Structure
Very simple, `model.py` is the model code, and `training_data.pth` is the path configuration file (I used it for checkpoint, basically storing values) that contains two important things: the weights and the attempts. You may ignore the other files.

## How to run it
1. Clone it
2. Open the folder in a code editor
3. Open the terminal
4. Run `python model.py`
5. Check the outputs

## What does the output mean?

- Epoch means how many times the computer has seen the data points; you can take it as how many attempts there are.
- Loss value is how close the output is to the actual answer; the bigger it is, the more inaccurate it is.
- The final answer... is the final answer the machine gives you, though not 100% accurate without sufficient attempts and data.
- Then it will print an object that contains the data it will store, including the weights, bias value and the attempts.

## // Skip the yapping if you know what machine learning is //

If you are also new to machine learning, the learning follows a very simple formula, and you have probably seen it in high school: `y = wx + b`
`w = weight, b = bias`

It is like the linear equation: `y = mx + b`

Although in this case, things might be a bit different because it uses a 4th-degree polynomial here. But the idea is similar.

So, on every attempt, the computer will adjust the weights and bias level, and try to make the y close to the value of the actual answer.

In this model, it follows the steps (In a loop): Forward Propagation (Feed data) > Loss Function (Find the gap) > Backpropagation (Find what was wrong) > Gradient (Improve in what direction) > Gradient Descent (Improve)

Notably, activation functions is skipped here because it will only make things worse in this model, as it cuts off the answers. You notice there are negative values in the training data, active functions make them zero, `max(x, 0)`.
