import { relu } from "./relu.js";
import { sigmoid } from "./sigmoid.js";

const Examples = {
  relu,
  sigmoid,
};

if (window) {
  Object.assign(window, { Examples });
}

export default Examples;
