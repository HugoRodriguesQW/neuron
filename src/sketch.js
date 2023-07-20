var training = false
var interactions = 0

function setup() {
  createCanvas(500, 500)
  background(255)

  neural = new NeuralNetwork(2, 6, 2, { learning_rate: 0.05 })

  dataset = {
    inputs: [
      [0, 1],
      [1, 0],
      [0, 0],
      [1, 1],
    ],
    outputs: [
      [0, 1],
      [1, 0],
      [0, 0],
      [0, 0],
    ],
  }
}

function draw() {
  if (training) {
    for (var i = 0; i < 10000; i++) {
      var index = Math.floor(Math.random() * dataset.inputs.length)
      neural.train(dataset.inputs[index], dataset.outputs[index])
    }

    if (neural.predict([0, 0])[0] < 0.04 && neural.predict([1, 0])[0] > 0.98) {
      training = false
      return console.info('terminou', JSON.parse(neural.export()))
    }

    console.log(
      'processing',
      neural.predict([0, 0])[0],
      neural.predict([1, 0])[0],
    )
  }

  document.addEventListener('click', () => {
    training = false
  })
}

function startLearn() {
  training = true
}
