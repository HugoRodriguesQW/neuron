import { Activation, NeuralNetwork } from "../src/brain.js";

// Sigmoid example: flee left or right based on danger direction

function setup() {
  const neural = new NeuralNetwork(2, 4, 2, {
    learning_rate: 0.012,
    activation: Activation.sigmoid,
  });

  const dataset = {
    inputs: [
      [1, 0], // Input 1: danger left
      [0, 1], // Input 2: danger right
      [0, 0], // Input 3: danger in the same position
      [1, 1], // Input 4: danger in both directions
    ],
    outputs: [
      [0, 1], // Output 1: Direction (right) required for Input 1
      [1, 0], // Output 2: Direction (left) required for Input 2
      [0, 0], // Output 3: Direction (none) required for Input 3
      [0, 0], // Output 4: Direction  (none) required for Input 4
    ],
  };

  let training = false;

  return {
    neural,
    dataset,
    stop: () => {
      training = false;
    },
    train: () => {
      training = true;
      const interval = setInterval(() => {
        if (!training) return clearInterval(interval);
        training = !interact({ neural, dataset }).learned;
      }, 1000 / 30);
    },
  };
}

var interactions = 0;

function interact({ neural, dataset }) {
  for (var i = 0; i < 1000; i++) {
    interactions++;
    var index = Math.floor(Math.random() * dataset.inputs.length);
    neural.train(dataset.inputs[index], dataset.outputs[index]);
  }

  const predictData = [
    neural.predict(dataset.inputs[0]),
    neural.predict(dataset.inputs[1]),
  ];

  if (
    predictData[0][1] > 0.96 &&
    predictData[0][0] < 0.04 &&
    predictData[1][1] < 0.04 &&
    predictData[1][0] > 0.96
  ) {
    console.info(
      `Learned with ${interactions} interactions: `,
      neural.export(),
      "use .neural to predict"
    );

    return { learned: true };
  }
  console.info("learning: ", predictData[0], predictData[1]);
  return { learned: false };
}

export const sigmoid = { interact, setup };
