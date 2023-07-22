import { Activation, NeuralNetwork } from "../src/brain.js";
import { Matrix } from "../src/matrix.js";

// ReLU example: identify numbers drawn on a 5x5 square (only those in the data-set)

function setup() {
  const neural = new NeuralNetwork(5 * 5, 10, 8, {
    learning_rate: 0.01,
    activation: Activation.relu,
  });

  const dataset = {
    inputs: [
      // Digit 0
      Matrix.flat([
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1],
      ]),
      // Digit 1
      Matrix.flat([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
      ]),

      // Digit 2
      Matrix.flat([
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1],
      ]),

      // Digit 3
      Matrix.flat([
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
      ]),
      // Digit 4
      Matrix.flat([
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
      ]),

      // Digit 5
      Matrix.flat([
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
      ]),

      // Digit 7
      Matrix.flat([
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
      ]),

      // Digit 9
      Matrix.flat([
        [0, 1, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1],
      ]),
    ],
    outputs: [
      [0, 0, 0, 0, 0, 0, 0, 0], // Digit 0
      [1, 0, 0, 0, 0, 0, 0, 0], // Digit 1
      [0, 1, 0, 0, 0, 0, 0, 0], // Digit 2
      [1, 1, 0, 0, 0, 0, 0, 0], // Digit 3
      [0, 0, 1, 0, 0, 0, 0, 0], // Digit 4
      [1, 0, 1, 0, 0, 0, 0, 0], // Digit 5
      [1, 1, 1, 0, 0, 0, 0, 0], // Digit 7
      [1, 0, 0, 1, 0, 0, 0, 0], // Digit 9
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
  for (var i = 0; i < 100; i++) {
    interactions++;
    var index = Math.floor(Math.random() * dataset.inputs.length);
    neural.train(dataset.inputs[index], dataset.outputs[index]);
  }

  const predictData = [
    neural.predict(Matrix.flat(dataset.inputs[0])),
    neural.predict(Matrix.flat(dataset.inputs[7])),
  ];

  if (
    predictData[0][7] < 0.2 &&
    predictData[0][0] < 0.2 &&
    predictData[1][3] > 0.8 &&
    predictData[1][0] > 0.8
  ) {
    console.info(
      `Learned with ${interactions} interactions: `,
      neural.export()
    );

    return { learned: true };
  }

  console.log("learning: ", predictData[0], predictData[1]);
  return { learned: false };
}

export const relu = { interact, setup };
