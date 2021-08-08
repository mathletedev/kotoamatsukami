import {
	add,
	dotMultiply,
	matrix,
	Matrix,
	multiply,
	random,
	subtract,
	transpose
} from "mathjs";
import { ACTIVATIONS, DERIVATIVES, NetworkFunction } from "./functions";

interface NetworkOptions {
	activationFunction?: NetworkFunction;
	derivative?: NetworkFunction;
	learningRate?: number;
}

export class Network {
	private weights: Matrix[];
	private biases: Matrix[];
	private activationFunction: NetworkFunction;
	private derivative: NetworkFunction;
	private lr: number;
	private data: Matrix[] = [];

	public constructor(
		layers: number[],
		{
			activationFunction = ACTIVATIONS.sigmoid,
			derivative = DERIVATIVES.sigmoid,
			learningRate = 0.5
		}: NetworkOptions = {}
	) {
		this.weights = layers
			.slice(0, -1)
			.map((layer, i) => random<Matrix>(matrix([layers[i + 1], layer]), -1, 1));
		this.biases = layers
			.slice(1)
			.map((layer) => random<Matrix>(matrix([layer, 1]), -1, 1));

		this.activationFunction = activationFunction;
		this.derivative = derivative;
		this.lr = learningRate;
	}

	private feedForward(inputs: number[]) {
		let current = transpose(matrix([inputs]));

		this.data = [current];
		for (let i = 0; i < this.weights.length; i++) {
			current = (
				add(multiply(this.weights[i], current), this.biases[i]) as Matrix
			).map(this.activationFunction);

			if (i < this.weights.length - 1) this.data.push(current);
		}

		return current;
	}

	private backPropagate(inputs: number[], targets: number[]) {
		const outputs = this.feedForward(inputs);

		let errors = subtract(matrix([targets]), outputs) as Matrix;
		let gradients = outputs.map(this.derivative);

		for (let i = this.weights.length - 1; i >= 0; i--) {
			gradients = dotMultiply(
				dotMultiply(gradients, errors),
				this.lr
			) as Matrix;
			const layer = this.data[i];

			this.weights[i] = add(
				this.weights[i],
				multiply(gradients, transpose(layer))
			) as Matrix;
			this.biases[i] = add(this.biases[i], gradients) as Matrix;

			errors = multiply(transpose(this.weights[i]), errors);
			gradients = layer.map(this.derivative);
		}
	}

	public train(
		inputs: number[][],
		targets: number[][],
		epochs: number,
		logging = false
	) {
		for (let i = 1; i <= epochs; i++) {
			if (logging && i % (epochs / 100) === 0)
				console.log(`Epoch [ ${i} / ${epochs} ]`);
			for (let x = 0; x < inputs.length; x++)
				this.backPropagate(inputs[x], targets[x]);
		}
	}

	public predict(inputs: number[]) {
		return this.feedForward(inputs).toArray()[0];
	}
}
