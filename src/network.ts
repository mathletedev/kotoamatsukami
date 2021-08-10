import { readFileSync, writeFileSync } from "fs";
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
		if (!layers.length) throw "Network must have at least 1 layer";

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
		const inputsLength = this.weights[0].size()[1];
		if (inputs.length !== inputsLength)
			throw `Inputs must be of length [ ${inputsLength} ]`;

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
		const targetsLength = this.weights[this.weights.length - 1].size()[0];
		if (targets.length !== targetsLength)
			throw `Targets must be of length [ ${targetsLength} ]`;

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

	public save(file: string) {
		if (!file.endsWith(".json")) throw "Save file must be of format [ JSON ]";

		const directory = `${process.cwd()}/${file}`;
		try {
			writeFileSync(
				directory,
				JSON.stringify({
					w: this.weights.map((weight) => weight.toArray()),
					b: this.biases.map((bias) => bias.toArray())
				}),
				{
					flag: "wx"
				}
			);
		} catch (err) {
			if ((err as Error).message.startsWith("EEXIST"))
				throw `File [ ${directory} ] already exists`;
		}
	}

	public load(file: string) {
		if (!file.endsWith(".json")) throw "Load file must be of format [ JSON ]";

		const directory = `${process.cwd()}/${file}`;
		try {
			const data = JSON.parse(readFileSync(directory).toString());

			if (!data.w?.length || !data.b?.length) throw "a";

			this.weights = data.w.map((weight: number[][]) => matrix(weight));
			this.biases = data.b.map((bias: number[][]) => matrix(bias));
		} catch (err) {
			if (err === "a") throw "Invalid load file data";
			if ((err as Error).message.startsWith("ENOENT"))
				throw `File [ ${directory} ] does not exist`;
			throw "Matrix sizes of load file data are invalid";
		}
	}
}
