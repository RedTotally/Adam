# Machine learning model for y = -17x^4 + x3 + 83x - 17

# Mistakes I have made: 

# 1. 
# Syntax errors, Python is more rigid compared to TypeScript.

# 2. 
# Since y = -17x^4 + x^3 + 83x - 17 is not a linear regression, and the model can't determine whether the data is linear, the loss value remained very high.
# e.g. loss value = 660,129,728.0, typically, the closer the value to 0, the better.
# In simple words, the gap between the xs is too big. If x = 5, then -17(5)^4 would be -10,625, and 83(5) would be 415. The machine tried to fit a linear model, y = mx + b, to each data point, resulting in a significant loss value.
# To solve this, I made four arrays, each representing a power, so the table contains every possible value from x^1 to x^4.
# Now the machine will stop thinking about how close two data points are, but instead, it discovers every possible data point from a big table.

# 3.
# y = -17(3.0)^4 + (3.0)3 + 83(3.0) - 17 = 1,136, not -1,280. Calculation mistake.

import torch
import torch.nn as nn
import torch.optim as optim
import os

attempt = 0

i_value = input("Enter a number: ")
i_value_number = float(i_value)

# I named it to Adam because this is the first machine learning code written by me.

class Adam(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.math = nn.Linear(4, 1)

    def forward(self, x): 
        return self.math(x)

model = Adam()

if os.path.exists("training_data.pth"):
    last_saved = torch.load("training_data.pth")
    model.load_state_dict(last_saved['weights'])
    model.eval()

    attempt = last_saved["attempts"]
else: 
    print("Cannot find the file.")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01) # This Adam is not my Adam lol it's a coincidence.

# Interesting thing I learned here, the X is uppercase because in math, uppercase letters refer to matrices and lowercase letters refer to vectors.

X_base = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]])
X = torch.cat([X_base, X_base**2, X_base**3, X_base**4], dim=1)

y = torch.tensor([[50.0], [-115.0], [-1136.0], [-3973.0], [-10102.0], [-21335.0], [-39910.0], [-68473.0], [-110078.0], [-168187.0]])

for epoch in range (100000): 
    attempt += 1
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10000 == 0:
       print(f"At epoch {epoch}, the loss value is {loss}.")

print(model(torch.tensor([[i_value_number, i_value_number**2, i_value_number**3, i_value_number**4]])).item())

store = {
    'weights': model.state_dict(),
    'attempts': attempt
}

torch.save(store, "training_data.pth")

print(torch.load("training_data.pth"))