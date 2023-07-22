import { Matrix } from "./matrix.js";

export class NeuralNetwork {
  constructor(i_nodes, h_nodes, o_nodes, { learning_rate, activation }) {
    this.i_nodes = i_nodes;
    this.h_nodes = h_nodes;
    this.o_nodes = o_nodes;

    this.bias_ih = new Matrix(h_nodes, 1);
    this.bias_ih.randomize();
    this.bias_ho = new Matrix(o_nodes, 1);
    this.bias_ho.randomize();

    //this.bias_ih.print("ih-bias");
    //this.bias_ho.print("ho-bias");

    this.weights_ih = new Matrix(h_nodes, i_nodes);
    this.weights_ih.randomize();

    this.weights_ho = new Matrix(o_nodes, h_nodes);
    this.weights_ho.randomize();

    //this.weights_ih.print("ih-weights");
    //this.weights_ho.print("ho-weights");

    this.learning_rate = learning_rate; //TODO: dynamic learning_rate
    this.activation = activation;
  }

  export() {
    const data = {
      i_nodes: this.i_nodes,
      h_nodes: this.h_nodes,
      o_nodes: this.o_nodes,

      bias_ih: this.bias_ih,
      bias_ho: this.bias_ho,

      weights_ih: this.weights_ih,
      weights_ho: this.weights_ho,

      learning_rate: this.learning_rate,
    };

    return data;
  }

  import(data) {
    Object.assign(this, data);
  }

  train(arr, target) {
    // INPUT => HIDDEN
    let input = Matrix.arrayToMatrix(arr);

    let hidden = Matrix.times(this.weights_ih, input);
    hidden = Matrix.add(hidden, this.bias_ih);
    hidden.map(this.activation);

    // HIDDEN => OUTPUT
    let output = Matrix.times(this.weights_ho, hidden);
    output = Matrix.add(output, this.bias_ho);
    output.map(this.activation);

    // OUTPUT => HIDDEN
    let expected = Matrix.arrayToMatrix(target);
    let output_error = Matrix.minus(expected, output);
    let d_output;

    if (Util.isSigmoid(this.activation)) {
      d_output = Matrix.map(output, Activation.dsigmoid);
    } else {
      // Apply the derivative of ReLU
      d_output = Matrix.map(output, (value) => (value >= 0 ? 1 : 0));
    }

    let hidden_transposed = Matrix.transpose(hidden);

    let gradient = Matrix.hadamard(output_error, d_output);
    gradient = Matrix.scalarTimes(gradient, this.learning_rate);

    // BIAS_HO CORRECTION
    this.bias_ho = Matrix.add(this.bias_ho, gradient);

    // WEIGHTS_HO CORRECTION
    let ho_delta = Matrix.times(gradient, hidden_transposed);
    this.weights_ho = Matrix.add(this.weights_ho, ho_delta);

    // HIDDEN => INPUT
    let weights_ho_transposed = Matrix.transpose(this.weights_ho);
    let hidden_error = Matrix.times(weights_ho_transposed, output_error);
    let d_hidden;

    if (Util.isSigmoid(this.activation)) {
      d_hidden = Matrix.map(hidden, Activation.dsigmoid);
    } else {
      // Apply the derivative of ReLU
      d_hidden = Matrix.map(hidden, (value) => (value >= 0 ? 1 : 0));
    }

    let input_transposed = Matrix.transpose(input);

    let gradient_hidden = Matrix.hadamard(hidden_error, d_hidden);
    gradient_hidden = Matrix.scalarTimes(gradient_hidden, this.learning_rate);

    // BIAS_HO CORRECTION
    this.bias_ih = Matrix.add(this.bias_ih, gradient_hidden);

    // WEIGHTS_IH CORRECTION
    let ih_delta = Matrix.times(gradient_hidden, input_transposed);
    this.weights_ih = Matrix.add(this.weights_ih, ih_delta);
  }

  predict(arr) {
    // INPUT -> HIDDEN
    let input = Matrix.arrayToMatrix(arr);

    let hidden = Matrix.times(this.weights_ih, input);
    hidden = Matrix.add(hidden, this.bias_ih);

    hidden.map(this.activation);

    // HIDDEN -> OUTPUT
    let output = Matrix.times(this.weights_ho, hidden);
    output = Matrix.add(output, this.bias_ho);
    output.map(this.activation);

    output = Matrix.matrixToArray(output);

    return output;
  }
}

export class Activation {
  static sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  static dsigmoid(x) {
    return x * (1 - x);
  }

  static relu(x) {
    return x >= 0 ? x : 0;
  }
}

class Util {
  static isSigmoid(func) {
    return func === Activation.sigmoid;
  }
}

if (window) {
  Object.assign(window, { Activation, NeuralNetwork, Util });
}
