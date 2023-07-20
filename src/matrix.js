class Matrix {
  // rows (length), cols (length)
  constructor(rows, cols) {
    this.rows = rows
    this.cols = cols
    this.data = []

    for (let i = 0; i < rows; i++) {
      let arr = []
      for (let j = 0; j < cols; j++) {
        arr.push(0) // TEMP: change later
      }

      this.data.push(arr)
    }
  }

  print(name) {
    console.group(name)
    console.table(this.data)
    console.groupEnd(name)
  }

  map(func) {
    this.data = this.data.map((arr, r) => {
      return arr.map((num, c) => {
        return func(num, r, c)
      })
    })

    return this
  }

  randomize() {
    this.map((elem, r, c) => {
      return Math.random() * 2 - 1 // -1 to 1
    })
  }

  static map(ref, func) {
    let matrix = new Matrix(ref.rows, ref.cols)
    matrix.data = ref.data.map((arr, r) => {
      return arr.map((num, c) => {
        return func(num, r, c)
      })
    })

    return matrix
  }

  static matrixToArray(matrix) {
    let arr = []
    matrix.map((elm, r, c) => {
      arr.push(elm)
    })

    return arr
  }

  static arrayToMatrix(arr) {
    let matrix = new Matrix(arr.length, 1)
    matrix.map((_, r) => {
      return arr[r]
    })
    return matrix
  }

  static add(A, B) {
    var matrix = new Matrix(A.rows, A.cols)

    matrix.map((num, i, j) => {
      return A.data[i][j] + B.data[i][j]
    })

    return matrix
  }

  static minus(A, B) {
    var matrix = new Matrix(A.rows, A.cols)

    matrix.map((num, i, j) => {
      return A.data[i][j] - B.data[i][j]
    })

    return matrix
  }

  static times(A, B) {
    var matrix = new Matrix(A.rows, B.cols)

    matrix.map((num, i, j) => {
      let sum = 0
      for (let k = 0; k < A.cols; k++) {
        let elm1 = A.data[i][k]
        let elm2 = B.data[k][j]
        sum += elm1 * elm2
      }

      return sum
    })

    return matrix
  }

  static hadamard(A, B) {
    var matrix = new Matrix(A.rows, A.cols)

    matrix.map((num, i, j) => {
      return A.data[i][j] * B.data[i][j]
    })

    return matrix
  }

  static scalarTimes(A, scalar) {
    var matrix = new Matrix(A.rows, A.cols)

    matrix.map((num, i, j) => {
      return A.data[i][j] * scalar
    })

    return matrix
  }

  static transpose(A) {
    var matrix = new Matrix(A.cols, A.rows)
    matrix.map((num, i, j) => {
      return A.data[j][i]
    })
    return matrix
  }
}
