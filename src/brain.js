// Feedforward e Backpropagation
// Entrada: um valor que será multiplicada e processada (number)
// Oculta: é o resultado da entrada * peso(e-o)
// Saída: é o resultado da oculta * peso(o-s)
// Camadas: Onde tudo acontece. Temos entrada - oculta - saída
// Nó (neuron): Cada camada é dividida em nós, que são interligados em matriz.
// Peso: Cada ligação tem um peso diferente.
// Ajuste: se a saída der 21.86 (saída) para 82.2 (entrada) e ele deveria
// ser 30 (saída), a IA vai ajustar os pesos para o resultado correto.
// O aprendizado é o ajuste dos pesos entre as camadas.
// Supervisionado: Predição através dos dados alimentados anteriormente
// Conceito: [Pesos de entrada] * [Entrada] = [Ocultas]
// Bias (tendencia): número arbitrário para acelerar o aprendizado,
// inputado pelo supervisor para entrada e para saída
// Função de Ativação: f(operação + tendência) usada na oculta e saída
// Sigmoid: tudo fica entre 0 e 1

class NeuralNetwork {
  constructor(i_nodes, h_nodes, o_nodes, { learning_rate }) {
    this.i_nodes = i_nodes
    this.h_nodes = h_nodes
    this.o_nodes = o_nodes

    this.bias_ih = new Matrix(h_nodes, 1)
    this.bias_ih.randomize()
    this.bias_ho = new Matrix(o_nodes, 1)
    this.bias_ho.randomize()

    //this.bias_ih.print("ih-bias");
    //this.bias_ho.print("ho-bias");

    this.weights_ih = new Matrix(h_nodes, i_nodes)
    this.weights_ih.randomize()

    this.weights_ho = new Matrix(o_nodes, h_nodes)
    this.weights_ho.randomize()

    //this.weights_ih.print("ih-weights");
    //this.weights_ho.print("ho-weights");

    this.learning_rate = learning_rate
  }

  export() {
    const data = JSON.stringify({
      i_nodes: this.i_nodes,
      h_nodes: this.h_nodes,
      o_nodes: this.o_nodes,

      bias_ih: this.bias_ih,
      bias_ho: this.bias_ho,

      weights_ih: this.weights_ih,
      weights_ho: this.weights_ho,

      learning_rate: this.learning_rate,
    })

    sessionStorage.setItem('neural', data)

    return data
  }

  import() {
    Object.assign(this, JSON.parse(sessionStorage.getItem('neural')))
  }

  train(arr, target) {
    // INPUT => HIDDEN
    let input = Matrix.arrayToMatrix(arr)

    let hidden = Matrix.times(this.weights_ih, input)
    hidden = Matrix.add(hidden, this.bias_ih)
    hidden.map(this.sigmoid)

    // HIDDEN => OUTPUT
    let output = Matrix.times(this.weights_ho, hidden)
    output = Matrix.add(output, this.bias_ho)
    output.map(this.sigmoid)

    // OUTPUT => HIDDEN
    let expected = Matrix.arrayToMatrix(target)
    let output_error = Matrix.minus(expected, output)
    let d_output = Matrix.map(output, this.dsigmoid)

    let hidden_transposed = Matrix.transpose(hidden)

    let gradient = Matrix.hadamard(output_error, d_output)
    gradient = Matrix.scalarTimes(gradient, this.learning_rate)

    // BIAS_HO CORRECTION
    this.bias_ho = Matrix.add(this.bias_ho, gradient)

    // WEIGHTS_HO CORRECTION
    let ho_delta = Matrix.times(gradient, hidden_transposed)
    this.weights_ho = Matrix.add(this.weights_ho, ho_delta)

    // HIDDEN => INPUT
    let weights_ho_transposed = Matrix.transpose(this.weights_ho)
    let hidden_error = Matrix.times(weights_ho_transposed, output_error)
    let d_hidden = Matrix.map(hidden, this.dsigmoid)
    let input_transposed = Matrix.transpose(input)

    let gradient_hidden = Matrix.hadamard(hidden_error, d_hidden)
    gradient_hidden = Matrix.scalarTimes(gradient_hidden, this.learning_rate)

    // BIAS_HO CORRECTION
    this.bias_ih = Matrix.add(this.bias_ih, gradient_hidden)

    // WEIGHTS_IH CORRECTION
    let ih_delta = Matrix.times(gradient_hidden, input_transposed)
    this.weights_ih = Matrix.add(this.weights_ih, ih_delta)
  }

  predict(arr) {
    // INPUT -> HIDDEN
    let input = Matrix.arrayToMatrix(arr)

    let hidden = Matrix.times(this.weights_ih, input)
    hidden = Matrix.add(hidden, this.bias_ih)

    hidden.map(this.sigmoid)

    // HIDDEN -> OUTPUT
    let output = Matrix.times(this.weights_ho, hidden)
    output = Matrix.add(output, this.bias_ho)
    output.map(this.sigmoid)

    output = Matrix.matrixToArray(output)

    return output
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x))
  }

  dsigmoid(x) {
    return x * (1 - x)
  }
}
