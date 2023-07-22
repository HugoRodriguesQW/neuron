
# neuron 🧠

### description

A feedforward neural network that can learn some simple tasks enveloped to be easily used and facilitate the evolution of generations.

---

For now, I'm working on the next steps, as it currently only has sigmoid as an activation function. I will add new function templates like ReLU for example. Also other gradients besides gradient descent. More hidden layers and extend your learning using selective evolutionary optimization.

---

### tests
 To test the neural network, go to [https://hugorodriguesqw.github.io/neuron/](https://hugorodriguesqw.github.io/neuron/) and open your browser console. It has already been imported into the window and can be used freely.
As it is still in early stage development, its use is quite restricted and annoying to use.

#### Create a new Neural Network
To create a new neural network, instantiate it using: **(input_count, hidden_count, output_count)**
```js
 const neural = new NeuralNetwork(25, 10, 8, {
    learning_rate: 0.01,
    activation: Activation.relu,
  });

// example from examples/relu.js

neural.train(input, expected)
neural.predict(input) 
```

#### Run Examples
Run the network setup below and start your workout: (relu or sigmoid for now)
```js
 const {
  neural, // neural network
  dataset, // dataset applied
  stop, // stop the training
  train, // start the training
 } = Examples.relu.setup()

```

Follow through the [issues](https://github.com/HugoRodriguesQW/neuron/issues)
