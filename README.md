<div align="center">
	<h1>
		k<img src="https://qph.fs.quoracdn.net/main-qimg-f611551d461bae7de8a32cc94cbe20ad" height="20em" />toamatsukami
	</h1>
	<p>
		<a href="https://www.npmjs.com/package/kotoamatsukami">
			<img src="https://img.shields.io/npm/v/kotoamatsukami.svg?maxAge=3600" />
		</a>
		<a href="https://www.npmjs.com/package/kotoamatsukami">
			<img src="https://img.shields.io/npm/dt/kotoamatsukami.svg?maxAge=3600" />
		</a>
	</p>
</div>

> A TypeScript ML library

## Installation

NPM

```bash
npm install kotoamatsukami
```

Yarn

```bash
yarn add kotoamatsukami
```

## Example

```ts
import { Network } from "kotoamatsukami";

const network = new Network([2, 3, 1]);

const inputs = [
	[0, 0],
	[1, 0],
	[0, 1],
	[1, 1]
];

const targets = [[0], [1], [1], [0]];

network.train(inputs, targets, 1000, true);

console.log(network.predict([0, 1]));
```

## License

[Apache License 2.0](https://github.com/mathletedev/kotoamatsukami/blob/main/LICENSE)
