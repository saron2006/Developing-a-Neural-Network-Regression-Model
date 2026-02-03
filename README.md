# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: SARON XAVIER A

### Register Number: 212223230197

```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x 


# Initialize the Model, Loss Function, and Optimizer



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
    optimizer.zero_grad()
    loss=criterion(ai_brain(X_train),y_train)
    loss.backward()
    optimizer.step()


    ai_brain.history['loss'].append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

```

### Dataset Information
<img width="754" height="676" alt="Screenshot 2026-02-02 094947" src="https://github.com/user-attachments/assets/99156218-bd6d-4489-b1be-8e393d6ef39b" />

### OUTPUT

### Training Loss Vs Iteration Plot
<img width="1069" height="685" alt="Screenshot 2026-02-02 094651" src="https://github.com/user-attachments/assets/45c3572b-4274-45d6-8fd0-e8b3483de70b" />
<img width="962" height="309" alt="Screenshot 2026-02-02 094547" src="https://github.com/user-attachments/assets/b430eac8-2dec-4c62-a6e6-9f44691afabd" />

### New Sample Data Prediction
<img width="855" height="125" alt="Screenshot 2026-02-02 094707" src="https://github.com/user-attachments/assets/0495a844-058d-428c-8b8b-de6296669ab9" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
