export type NetworkFunction = (x: number) => number;

export const ACTIVATIONS: Record<string, NetworkFunction> = {
	sigmoid: (x) => Math.exp(x) / (Math.exp(x) + 1),
	tanh: (x) => Math.tanh(x),
	relu: (x) => Math.max(0, x)
};

export const DERIVATIVES: Record<string, NetworkFunction> = {
	sigmoid: (x) => x * (1 - x),
	tanh: (x) => 1 - x ** 2,
	relu: (x) => (x > 0 ? 1 : 0)
};
